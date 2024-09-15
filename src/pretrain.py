
"""Script for training JEPA model"""

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

from src.masks.random import MaskCollator as RandMaskCollator
from src.masks.multiblock import MaskCollator as MBMaskCollator
from src.masks.utils import apply_masks
from src.utils.distributed import (
    init_distributed,
    AllReduce
)
from src.utils.logging import (
    CSVLogger,
    gpu_timer,
    grad_logger,
    AverageMeter)
from src.utils.tensors import repeat_interleave_batch
from src.datasets.util import make_dataset, worker_init_fn
from tensorboardX import SummaryWriter
import lmdb

from src.helper import (
    load_checkpoint,
    init_model,
    init_opt)
from src.transforms import make_transforms

log_timings= True
log_freq = 10
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
    load_model = args['meta']['load_checkpoint'] or resume_preempt
    r_file = args['meta']['read_checkpoint']
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']
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
    
    # Mask settings
    allow_overlap = args['mask']['allow_overlap']  # whether to allow overlap b/w context and target blocks
    patch_size = args['mask']['patch_size']  
    num_enc_masks = args['mask']['num_enc_masks']  # number of context blocks
    min_keep = args['mask']['min_keep']  # minimum number of patches to keep for context block
    enc_mask_scale = args['mask']['enc_mask_scale']  # scale of context blocks
    num_pred_masks = args['mask']['num_pred_masks']  # number of target blocks
    pred_mask_scale = args['mask']['pred_mask_scale']  # scale of target blocks
    aspect_ratio = args['mask']['aspect_ratio']  # aspect ratio of target blocks
    
    # Training settings
    ema = args['optimization']['ema']
    ipe_scale = args['optimization']['ipe_scale']
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
    dump = os.path.join(folder, 'params-ijepa.yaml')
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(dump, 'w') as f:
        yaml.dump(args, f)
    
    
    try:
        mp.set_start_method('spawn')
    except Exception:
        pass
    
    # Initialize distributed training
    world_size, rank = init_distributed()
    logger.info(f"Initalized distributed training with world size {world_size} and rank {rank}")
    if rank > 0:
        logger.setLevel(logging.ERROR)
        
    # log/checkpointing paths
    log_file = os.path.join(folder, f"{tag}_r{rank}.csv")
    save_path = os.path.join(folder, f"{tag}" + "-ep{epoch}.pth.tar")
    latest_path = os.path.join(folder, f"{tag}-latest.pth.tar")
    load_path = None
    if load_model:
        load_path = os.path.join(folder, r_file) if r_file is not None else latest_path
    
    # make csv logger
    csv_logger = CSVLogger(log_file,
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.5f', 'loss'),
                           ('%.5f', 'mask-A'),
                           ('%.5f', 'mask-B'),
                           ('%d', 'time (ms)'))
    
    # initialize model
    encoder, predictor = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name)
    target_encoder = copy.deepcopy(encoder)
    
    # make transforms
    mask_collator  = MBMaskCollator(
        input_size=crop_size,
        patch_size=patch_size,
        pred_mask_scale=pred_mask_scale,
        enc_mask_scale=enc_mask_scale,
        aspect_ratio=aspect_ratio,
        nenc=num_enc_masks,
        npred=num_pred_masks,
        allow_overlap=allow_overlap,
        min_keep=min_keep)
    
    transforms = make_transforms(
        crop_size=crop_size,
        crop_scale=crop_scale,
        gaussian_blur=use_gaussian_blur,
        horizontal_flip=use_horizontal_flip,
        color_distortion=use_color_distortion,
        color_jitter=color_jitter)
    
    # init dataloaders 
    train_dataset = make_dataset(
        dataset_name,
        lmdb_path is not None,
        root=root_path,
        transform=transforms,
        train=True,
    )
    test_dataset = make_dataset(
        dataset_name,
        lmdb_path is not None,
        root=root_path,
        transform=transforms,
        train=False
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
        collate_fn=mask_collator,
        sampler=train_sampler,
        batch_size=batch_size,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False,
        prefetch_factor=prefetch_factor,
        drop_last=True,
        worker_init_fn=worker_init_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        collate_fn=mask_collator,
        sampler=test_sampler,
        batch_size=batch_size,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False,
        prefetch_factor=prefetch_factor,
        drop_last=True,
        worker_init_fn=worker_init_fn
    )
    ipe = len(train_loader)
    
    # init optimizer
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        use_bfloat16=use_bfloat16)
    encoder = DistributedDataParallel(encoder, static_graph=True)
    predictor = DistributedDataParallel(predictor, static_graph=True)
    target_encoder = DistributedDataParallel(target_encoder)
    
    # Freeze target encoder
    for p in target_encoder.parameters():
        p.requires_grad = False
        
    momentum_scheduler = (
        ema[0] + i * (ema[1] - ema[0]) / (num_epochs * ipe * ipe_scale)
        for i in range(int(ipe*num_epochs*ipe_scale) + 1)
    )
    
    start_epoch = 0
    # load training checkpoint
    if load_model:
        encoder, predictor, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler)
        # set scheduler to start_epoch
        for _ in range(start_epoch*ipe):
            scheduler.step()
            wd_scheduler.step()
            next(momentum_scheduler)
            mask_collator.step()
    
    
    def save_checkpoint(epoch, loss_meter):
        save_dict = {
            "encoder": encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "target_encoder": target_encoder.state_dict(),
            "opt": optimizer.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "epoch": epoch,
            "loss": loss_meter.avg,
            "batch_size": batch_size,
            "world_size": world_size,
            "lr": lr
        }
        if rank == 0:
            torch.save(save_dict, latest_path)
            if (epoch + 1) % checkpoint_freq == 0:
                torch.save(save_dict, save_path.format(epoch=f"{epoch + 1}"))
    
    # training loop
    for epoch in range(start_epoch, num_epochs):
        logging.info(f"Starting epoch {epoch + 1}")
        logging.info(f"Number of iterations per epoch: {ipe}")
        train_sampler.set_epoch(epoch)
        
        loss_meter = AverageMeter()
        pred_mask_meter = AverageMeter()  # meter for number of patches in target block
        enc_mask_meter = AverageMeter()  # meter for number of patches in context block
        inf_time_meter = AverageMeter()  # meter for inference time
        data_time_meter = AverageMeter()  # meter for data loading time
        
        s = time.time()
        for itr, (udata, masks_enc, masks_pred) in enumerate(train_loader):
            
            def load_imgs():
                imgs = udata['image'].to(device, non_blocking=True)
                masks_1 = [m.to(device, non_blocking=True) for m in masks_enc]  # [(B, min_keep_enc), (B, min_keep_enc), ...]
                masks_2 = [m.to(device, non_blocking=True) for m in masks_pred]   # [(B, min_keep_pred), (B, min_keep_pred), ...]
                return imgs, masks_1, masks_2
            imgs, masks_enc, masks_pred = load_imgs()
            elapsed_ms = (time.time() - s) * 1000
            data_time_meter.update(elapsed_ms)
            pred_mask_meter.update(len(masks_enc[0][0]))
            enc_mask_meter.update(len(masks_pred[0][0]))
            
            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()
                
                def forward_target():
                    with torch.no_grad():
                        h = target_encoder(imgs)  # (B, N, D)
                        h = F.layer_norm(h, (h.size(-1), ))  
                        B = len(h)
                        
                        h = apply_masks(h, masks_pred)  # (n_pred * B, min_keep_pred, D)
                        h = repeat_interleave_batch(h, B, repeat=len(masks_enc))  # (n_pred * n_enc * B, min_keep_pred, D)
                        return h

                def forward_context():
                    z = encoder(imgs, masks_enc)  # (n_enc * B, min_keep_enc, D)
                    z = predictor(z, masks_enc, masks_pred)  # (n_pred * n_enc * B, min_keep_pred, D)
                    return z
                
                def loss_fn(z, h):
                    loss = F.smooth_l1_loss(z, h)
                    loss = AllReduce.apply(loss)  # average loss across all GPUs
                    return loss

                # forward pass
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
                    h = forward_target()
                    z = forward_context()
                    loss = loss_fn(z, h)
                
                # backward pass
                if use_bfloat16:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update() 
                else:
                    loss.backward()
                    optimizer.step()
                grad_stats = grad_logger(encoder.named_parameters())
                optimizer.zero_grad()
                
                # Update target encoder using momentum
                with torch.no_grad():
                    m = next(momentum_scheduler)
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        param_k.data.mul_(m).add_(param_q.detach().data, alpha=1-m)
                return loss.item(), _new_lr, _new_wd, grad_stats
            
            results, etime = gpu_timer(train_step)
            loss, _new_lr, _new_wd, grad_stats = results
            loss_meter.update(loss)
            inf_time_meter.update(etime)
            
            def log_stats():
                csv_logger.log(epoch + 1, itr, loss, pred_mask_meter.val, enc_mask_meter.val, etime)
                writer.add_scalar('Loss/train', loss_meter.val, itr + epoch * ipe)
                writer.add_scalar('LearningRate', _new_lr, itr + epoch * ipe)
                writer.add_scalar('WeightDecay', _new_wd, itr + epoch * ipe)
                writer.add_scalar('Time/InferenceTime', inf_time_meter.val, itr + epoch * ipe)
                writer.add_scalar('Time/DataTime', data_time_meter.val, itr + epoch * ipe)
                writer.add_scalar('Masks/PredMask', pred_mask_meter.val, itr + epoch * ipe)
                writer.add_scalar('Masks/EncMask', enc_mask_meter.val, itr + epoch * ipe)
                
                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    logger.info('[%d, %5d] loss: %.3f '
                                'masks: (context) %.1f (target) %.1f '
                                '[wd: %.2e] [lr: %.2e] '
                                '[mem: %.2e] '
                                '[model: (%.1f ms)]'
                                '[data: (%.1f ms)]'
                                % (epoch + 1, itr,
                                   loss_meter.avg,
                                   pred_mask_meter.avg,
                                   enc_mask_meter.avg,
                                   _new_wd,
                                   _new_lr,
                                   torch.cuda.max_memory_allocated() / 1024.**2,
                                   inf_time_meter.avg,
                                   data_time_meter.avg))
                
                if grad_stats is not None:
                    logger.info('[%d, %5d] grad_stats: [%.2e %.2e] (%.2e, %.2e)'
                                % (epoch + 1, itr,
                                    grad_stats.first_layer,
                                    grad_stats.last_layer,
                                    grad_stats.min,
                                    grad_stats.max))
                    writer.add_scalar('GradStats/FirstLayer', grad_stats.first_layer, itr + epoch * ipe)
                    writer.add_scalar('GradStats/LastLayer', grad_stats.last_layer, itr + epoch * ipe)
                    writer.add_scalar('GradStats/Min', grad_stats.min, itr + epoch * ipe)
                    writer.add_scalar('GradStats/Max', grad_stats.max, itr + epoch * ipe)
                        
            log_stats()
            assert not np.isnan(loss), "Loss is NaN"
            
            s = time.time()
            
        # save checkpoint
        logger.info(f"Saving checkpoint for epoch {epoch + 1}")
        save_checkpoint(epoch+1, loss_meter)        
    
    logger.info("Training complete")
    logging.info(f"To resume training, run: python train.py --resume --r_file {latest_path}")
    logging.info(f"To check tensorboard logs, run: tensorboard --logdir={folder}")
    writer.close()               

if __name__ == '__main__':
    main()
        