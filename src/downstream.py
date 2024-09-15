
"""Script to train JEPA on downstream tasks"""

import os

import torch.distributed

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import time
import copy
import logging
import sys
import yaml

import numpy as np

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from src.utils.distributed import (
    init_distributed,
    AllReduce,
    AllReduceSum,
)
from src.utils.logging import (
    CSVLogger,
    gpu_timer,
    grad_logger,
    AverageMeter)
from src.datasets.util import make_dataset, worker_init_fn
from tensorboardX import SummaryWriter

from src.helper import (
    load_checkpoint_for_downstream,
    downstream_setup,
    init_opt_for_downstream,
    init_model_for_downstream,
    )
from src.transforms import make_transforms

log_timings = True
log_freq = 1
checkpoint_freq = 50

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

def main(args, resume_preempt=False):
    use_bfloat16 = args['meta']['use_bfloat16']
    dataset_name = args['meta']['dataset_name']
    model_name = args['meta']['model_name']
    load_pretrained_model = args['meta']['load_checkpoint'] or resume_preempt
    r_file = args['meta']['read_checkpoint']
    strategy = args['meta']['strategy']  # linear, full
    
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
    
    # Data settings
    use_gaussian_blur = args['data']['use_gaussian_blur']
    use_horizontal_flip = args['data']['use_horizontal_flip']
    use_color_distortion = args['data']['use_color_distortion']
    color_jitter = args['data']['color_jitter_strength']
    batch_size = args['data']['batch_size']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    prefetch_factor = args['data']['prefetch_factor']
    root_path = args['data']['root_path']
    lmdb_path = args['data']['lmdb_path']
    crop_size = args['data']['crop_size']
    crop_scale = args['data']['crop_scale']
    
    # Model settings
    patch_size = args['model']['patch_size']
    
    # Training settings
    wd = float(args['optimization']['weight_decay'])
    final_wd = float(args['optimization']['final_weight_decay'])
    num_epochs = args['optimization']['epochs']
    warmup = args['optimization']['warmup']
    start_lr = args['optimization']['start_lr']
    lr = args['optimization']['lr']
    final_lr = args['optimization']['final_lr']

    # Logging settings
    folder = args['logging']['folder']
    tag = args['logging']['write_tag']
    writer = SummaryWriter(log_dir=folder)
    
    # Save args
    dump = os.path.join(folder, 'params-downstream-ijepa.yaml')
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(dump, 'w') as f:
        yaml.dump(args, f)
    
    try:
        mp.set_start_method('spawn', force=True)
    except Exception:
        pass
    
    # Initialize distributed training
    world_size, rank = init_distributed()
    logging.info(f"Running... (rank: {rank}/{world_size})")
    if rank > 0:
        logger.setLevel(logging.ERROR) 
    
    # log/checkpoint paths
    log_file = os.path.join(folder, f"{tag}_r{rank}.csv")
    save_path = os.path.join(folder, f"{tag}" + "-ep{epoch}.pth.tar")
    latest_path = os.path.join(folder, f"{tag}-latest.pth.tar")
    load_path = None
    if load_pretrained_model:
        load_path = os.path.join(r_file)
        
    # csv logger
    csv_logger = CSVLogger(
        log_file,
        ('%d', 'epoch'),
        ('%d', 'itr'),
        ('%.5f', 'loss'),
        ('%d', 'time (ms)')
    )
    
    # initialize model
    encoder = init_model_for_downstream(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        model_name=model_name,
    )
    transform = make_transforms(
        crop_size=crop_size,
        crop_scale=crop_scale,
        gaussian_blur=use_gaussian_blur,
        horizontal_flip=use_horizontal_flip,
        color_distortion=use_color_distortion,
        color_jitter=color_jitter,
    )
    
    # init dataloaders
    train_dataset = make_dataset(
        dataset_name,
        lmdb_path is not None,
        root=root_path,
        transform=transform,
        download=True,
        train=True,
    )
    test_dataset = make_dataset(
        dataset_name,
        lmdb_path is not None,
        root=root_path,
        transform=transform,
        download=True,
        train=False,
    )
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=train_dataset,
        num_replicas=world_size,
        rank=rank)
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=test_dataset,
        num_replicas=world_size,
        rank=rank)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=batch_size,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        sampler=test_sampler,
        batch_size=batch_size,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )
    ipe = len(train_loader)
    
    # Note: 
    # (For Linear probing)
    # We should freeze the encoder weights and only train the linear layer for downstream tasks. 
    # (For Full finetuning)
    # We should train the entire model, including the encoder and the linear layer.
    # (For both cases)
    # Linear layer takes the encoder global averaged pooled features as input. We pool last 4 layers or last layer's hidden states.
    # The addition of the linear layer must be done before the optimizer initialization.
    encoder = downstream_setup(
        encoder=encoder,
        num_classes=train_dataset.num_classes,
        strategy=strategy,
        device=device
    )
    encoder = DistributedDataParallel(encoder)
    
    start_epoch = 0
    if load_pretrained_model:
        encoder = load_checkpoint_for_downstream(device, load_path, encoder) 
        logging.info(f"Checkpoint loaded from {load_path}", extra={'color': 'green'})
    else:
        # colored log
        logging.info("No checkpoint loaded, starting from scratch.", extra={'color': 'orange'})
    
    # optimizer setup
    optimizer, scaler, scheduler, wd_scheduler, encoder = init_opt_for_downstream(
        encoder=encoder,
        strategy=strategy,
        iterations_per_epoch=ipe,
        start_lr=start_lr,
        ref_lr=lr,
        warmup=warmup,
        num_epochs=num_epochs,
        wd=wd,
        final_wd=final_wd,
        final_lr=final_lr,
        use_bfloat16=use_bfloat16,
    )
    
    def save_checkpoint(epoch, loss_meter):
        save_dict = {
            'epoch': epoch,
            'target_encoder': encoder.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr,
            'loss': loss_meter.avg
        }
        if rank == 0:
            torch.save(save_dict, latest_path)
            if (epoch + 1) % checkpoint_freq == 0:
                torch.save(save_dict, save_path.format(epoch=f"{epoch+1}"))
    
    # training loop
    logging.info(f"Iterations per epoch: {ipe}")
    for epoch in range(start_epoch, num_epochs):
        logging.info(f"Starting epoch {epoch + 1}/{num_epochs}")
        train_sampler.set_epoch(epoch)
        
        loss_meter = AverageMeter()
        inf_time_meter = AverageMeter()
        data_time_meter = AverageMeter()
        
        s = time.time()
        correct_sum = 0
        for itr, batch in enumerate(train_loader):
            
            def load_imgs():
                imgs = batch['image'].to(device, non_blocking=True)
                labels = batch['label'].to(device, non_blocking=True)
                return imgs, labels
            imgs, labels = load_imgs()
            elapsed = time.time() - s
            data_time_meter.update(elapsed)
            
            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()
                
                # forward pass
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
                    if strategy == "linear":
                        with torch.no_grad():
                            enc_features = encoder(imgs)
                    elif strategy == "full":
                        enc_features = encoder(imgs)  # (B, N, D)
                    pooler_features = torch.mean(enc_features, dim=1)  # (B, D)
                    logits = encoder.module.head(pooler_features)  # (B, K)
                    loss = F.cross_entropy(logits, labels)  
                    loss = AllReduce.apply(loss)  # average loss across all processes
                    preds = torch.argmax(logits, dim=1)  # (B)
                    correct = torch.sum(preds == labels)
                    correct = AllReduce.apply(correct)  # average correct predictions across all processes
                    
                # backward pass
                if use_bfloat16:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                optimizer.zero_grad()
                
                return loss.item(), correct.item(), _new_lr, _new_wd
            
            results, etime = gpu_timer(train_step)
            loss, correct, _new_lr, _new_wd = results
            correct_sum += correct
            loss_meter.update(loss)
            inf_time_meter.update(etime)
            
            def log_stats():
                csv_logger.log(epoch + 1, itr, loss, etime)
                writer.add_scalar('Loss/train', loss_meter.val, itr + epoch * ipe)
                writer.add_scalar('Accuracy/train', correct_sum / (batch_size * (itr + 1)), itr + epoch * ipe)
                writer.add_scalar('LearningRate', _new_lr, itr + epoch * ipe)
                writer.add_scalar('WeightDecay', _new_wd, itr + epoch * ipe)
                writer.add_scalar('Time/InferenceTime', inf_time_meter.val, itr + epoch * ipe)
                writer.add_scalar('Time/DataTime', data_time_meter.val, itr + epoch * ipe)
                writer.flush()
            
                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    logger.info('[%d, %5d] loss: %.3f '
                                '[acc: %.3f]'
                                '[wd: %.2e] [lr: %.2e] '
                                '[mem: %.2e] '
                                '[model: (%.1f ms)]'
                                '[data: (%.1f ms)]'
                                % (epoch + 1, itr,
                                   loss_meter.avg,
                                   correct_sum / (batch_size * (itr + 1)),
                                   _new_wd,
                                   _new_lr,
                                   torch.cuda.max_memory_allocated() / 1024.**2,
                                   inf_time_meter.avg,
                                   data_time_meter.avg))
                
            log_stats()
            assert not np.isnan(loss), "Loss is NaN"
            
            s = time.time()
        
        # save checkpoint
        logging.info(f"Epoch {epoch + 1}/{num_epochs} completed, saving checkpoint...")
        save_checkpoint(epoch, loss_meter)
    
    logging.info("Training completed.")
    logging.info(f"To check tensorboard logs, run: tensorboard --logdir={folder}")
    writer.close()

if __name__ == "__main__":
    main()
                        
                    
                        
                    
               
    
    