"""Script to evaluate JEPA on downstream tasks"""

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

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

def main(args, resume_preempt=False):
    dataset_name = args['meta']['dataset_name']
    model_name = args['meta']['model_name']
    load_path = args['meta']['read_checkpoint']
    
    assert load_path is not None, "Must provide a checkpoint to evaluate"
    
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
    
    # Data settings
    batch_size = args['data']['batch_size']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    prefetch_factor = args['data']['prefetch_factor']
    crop_size = args['data']['crop_size']
    root_path = args['data']['root_path']
    lmdb_path = args['data']['lmdb_path']

    # Model settings
    patch_size = args['model']['patch_size']
    
    # Logging settings
    folder = args['logging']['folder']
    tag = args['logging']['write_tag']
    writer = SummaryWriter(log_dir=folder)
    
    # save config
    dump = os.path.join(folder, 'params-evaluation-ijepa.yaml')
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(dump, 'w') as f:
        yaml.dump(args, f)

    try:
        mp.set_start_method('spawn', force=True)
    except Exception:
        pass
    
    # Initialize distributed inference
    world_size, rank = init_distributed()
    logging.info(f"Running... (rank: {rank}/{world_size})")
    if rank > 0:
        logger.setLevel(logging.ERROR) 

    # initialize model
    encoder = init_model_for_downstream(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        model_name=model_name,
    )
    
    # init transforms
    transform = make_transforms(
        crop_size=crop_size,
        crop_scale=(1.0, 1.0),
        color_distortion=False,
        gaussian_blur=False,
        horizontal_flip=False,
    )
    
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
        persistent_workers=False,
        prefetch_factor=prefetch_factor,
        drop_last=False,
        worker_init_fn=worker_init_fn,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        sampler=test_sampler,
        batch_size=batch_size,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False,
        prefetch_factor=prefetch_factor,
        drop_last=False,
        worker_init_fn=worker_init_fn,
    )
    ipe = len(train_loader)
    num_samples_train = len(train_dataset)
    num_samples_test = len(test_dataset)

    encoder = downstream_setup(
        encoder=encoder,
        num_classes=train_dataset.num_classes,
        strategy="evaluation",
        device=device,
    )
    encoder = DistributedDataParallel(encoder)
    
    # Load checkpoint
    encoder = load_checkpoint_for_downstream(device, load_path, encoder)
    
    # Evaluation on training set
    encoder.eval()
    train_loss = 0.
    train_correct = 0
    train_loader.sampler.set_epoch(0)
    with torch.no_grad():
        for i, data in enumerate(train_loader):
            images = data['image'].to(device)
            labels = data['label'].to(device)
            enc_features = encoder(images)  # (B, N, D)
            pooler_features = torch.mean(enc_features, dim=1)  # (B, D)
            logits = encoder.module.head(pooler_features)  # (B, K)
            loss = F.cross_entropy(logits, labels)
            loss = AllReduce.apply(loss)  # average loss across all processes
            train_loss += loss.item()
            _, predicted = logits.max(1)
            correct = predicted.eq(labels).sum()
            correct = AllReduceSum.apply(correct)  # sum correct predictions across all processes
            train_correct += correct.item()
    
    train_acc = 100. * train_correct / num_samples_train
    train_loss = train_loss / ipe
    logging.info(f"Train Acc: {train_acc:.2f}, Train Loss: {train_loss:.4f}")
    writer.add_scalar('Train/Acc', train_acc, 0)
    writer.add_scalar('Train/Loss', train_loss, 0)
    
    # Evaluation on test set
    test_loss = 0.
    test_correct = 0
    test_loader.sampler.set_epoch(0)
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            images = data['image'].to(device)
            labels = data['label'].to(device)
            enc_features = encoder(images)  # (B, N, D)
            pooler_features = torch.mean(enc_features, dim=1)
            logits = encoder.module.head(pooler_features)
            loss = F.cross_entropy(logits, labels)
            loss = AllReduce.apply(loss)  # average loss across all processes
            
            test_loss += loss.item()
            _, predicted = logits.max(1)
            correct = predicted.eq(labels).sum()
            correct = AllReduceSum.apply(correct)  # sum correct predictions across all processes
            test_correct += correct.item()
    
    test_acc = 100. * test_correct / num_samples_test
    test_loss = test_loss / ipe
    writer.add_scalar('Test/Acc', test_acc, 0)
    writer.add_scalar('Test/Loss', test_loss, 0)
    logging.info(f"Test Acc: {test_acc:.2f}, Test Loss: {test_loss:.4f}")
    
    writer.close()
    logging.info(f"Finished evaluation. Results saved to {folder}")
    logging.info(f"Tensorboard results can be viewed by running `tensorboard --logdir={folder}`")
    return 
    
    
    