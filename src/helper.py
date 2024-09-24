import logging
import sys

import torch

import src.models.vision_transformer as vit
from src.models.vision_transformer import VisionTransformer, VisionTransformerPredictor
from src.utils.schedulers import (
    WarmupCosineSchedule,
    LinearDecaySchedule,
    CosineWDSchedule)
from src.utils.tensors import trunc_normal_

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

def load_checkpoint_for_downstream(
    device,
    r_path,
    target_encoder
) -> VisionTransformer:
    try:
        checkpoint = torch.load(r_path, map_location=torch.device('cpu'), weights_only=True)
        epoch = checkpoint['epoch']
        
        # -- loading target_encoder
        logging.info(f'loaded pretrained target encoder from epoch {epoch}')
        logging.info(f"Checkpoint keys: {list(checkpoint.keys())}")
        pretrained_dict = checkpoint['target_encoder']
        msg = target_encoder.load_state_dict(pretrained_dict)
        logger.info(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')
        
        del checkpoint
        
    except Exception as e:
        logger.info(f'Encountered exception when loading checkpoint {e}')
        
    return target_encoder

def load_checkpoint(
    device,
    r_path,
    encoder, 
    predictor,
    target_encoder,
    opt,
    scaler,
):
    try:
        checkpoint = torch.load(r_path, map_location=torch.device('cpu'), weights_only=True)
        epoch = checkpoint['epoch']
        
        # -- loading encoder
        pretrained_dict = checkpoint['encoder']
        msg = encoder.load_state_dict(pretrained_dict)
        logger.info(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')
        
        # -- loading predictor
        pretrained_dict = checkpoint['predictor']
        msg = predictor.load_state_dict(pretrained_dict)
        logger.info(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')
        
        # -- loading target_encoder
        if target_encoder is not None:
            print(list(checkpoint.keys()))
            pretrained_dict = checkpoint['target_encoder']
            msg = target_encoder.load_state_dict(pretrained_dict)
            logger.info(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')
        
        # -- loading optimizer
        opt.load_state_dict(checkpoint['opt'])
        if scaler is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        logger.info(f'loaded optimizers from epoch {epoch}')
        logger.info(f'read-path: {r_path}')
        del checkpoint
        
    except Exception as e:
        logger.info(f'Encountered exception when loading checkpoint {e}')
        epoch = 0
        
    return encoder, predictor, target_encoder, opt, scaler, epoch

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, torch.nn.Linear) and m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, torch.nn.LayerNorm):
        torch.nn.init.ones_(m.weight)
        torch.nn.init.zeros_(m.bias)
        
def init_model(
    device, 
    patch_size=16,
    model_name="vit_base",
    crop_size=224,
    pred_depth=6,
    pred_emb_dim=384,
):
    encoder = vit.__dict__[model_name](
        img_size=[crop_size],
        patch_size=patch_size,
    )
    for m in encoder.modules():
        init_weights(m)
    encoder.to(device)
    logger.info(encoder)
    
    predictor = vit.__dict__["vit_predictor"](
        num_patches=encoder.patch_embed.num_patches,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=pred_emb_dim,
        depth=pred_depth,
        num_heads=encoder.num_heads,
    )
        
    for m in predictor.modules():
        init_weights(m)
    
    predictor.to(device)
    return encoder, predictor

def init_opt(
    encoder,
    predictor,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.,
    use_bfloat16=False,
    ipe_scale=1.25,
    scheduler_type="cosine"
):
    # exclude bias and layernorm parameters from weight decay
    param_groups = [
        {
            'params': (p for n, p in encoder.named_parameters() if ('bias' not in n) and (len(p.shape) != 1)),
        }, {
            'params': (p for n, p in predictor.named_parameters() if ('bias' not in n) and (len(p.shape) != 1)),  
        }, {
            'params': (p for n, p in encoder.named_parameters() if ('bias' in n) or (len(p.shape) == 1)),
            'weight_decay': 0.,
            'WD_exclude': True,
        }, {
            'params': (p for n, p in predictor.named_parameters() if ('bias' in n) or (len(p.shape) == 1)),
            'weight_decay': 0.,
            'WD_exclude': True,
        }
    ]
    
    logger.info('Using AdamW')
    optimizer = torch.optim.AdamW(param_groups)
    if scheduler_type == "cosine":
        scheduler = WarmupCosineSchedule(
            optimizer,
            warmup_steps=int(warmup * iterations_per_epoch),
            start_lr=start_lr,
            ref_lr=ref_lr,
            final_lr=final_lr,
            T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
        )
    elif scheduler_type == "linear":
        scheduler = LinearDecaySchedule(
            optimizer,
            start_lr=start_lr,
            final_lr=final_lr,
            T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
        )
    else:
        raise ValueError(f"Scheduler type {scheduler_type} not recognized")
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_bfloat16) if use_bfloat16 else None
    return optimizer, scaler, scheduler, wd_scheduler

def downstream_setup(
    encoder: VisionTransformer, 
    num_classes: int, 
    strategy: str, 
    device: torch.device = "cuda:0"
):
    if strategy == "linear":
        # Freeze the encoder
        for param in encoder.parameters():
            param.requires_grad = False
        # Add a linear layer
        encoder.head = torch.nn.Linear(encoder.embed_dim, num_classes)
        encoder.head.to(device)
    elif strategy == "full":
        encoder.head = torch.nn.Linear(encoder.embed_dim, num_classes)
        encoder.head.to(device)
    elif strategy == "evaluation":
        # For DDP evaluation, we need to leave the part of encoder trainable, but we don't update it's parameters
        for param in encoder.parameters():
            param.requires_grad = False
        encoder.head = torch.nn.Linear(encoder.embed_dim, num_classes)
        encoder.head.to(device)
    else:
        raise ValueError(f"Strategy {strategy} not recognized")
    return encoder

def init_model_for_downstream(
    device,
    patch_size=16,
    model_name="vit_base",
    crop_size=224,
):
    encoder = vit.__dict__[model_name](
        img_size=[crop_size],
        patch_size=patch_size,
    )
    for m in encoder.modules():
        init_weights(m)
    encoder.to(device)
    logger.info(encoder)
    
    return encoder

def init_opt_for_downstream(
    encoder: torch.nn.Module, 
    strategy: str, 
    iterations_per_epoch: int,
    start_lr: float,
    ref_lr: float,
    warmup: float,
    num_epochs: int,
    wd: float=1e-6,
    final_wd: float=1e-6,
    final_lr: float=0.,
    use_bfloat16: bool=False,
    ipe_scale: float=1.25,
    scheduler_type: str="cosine"
):
    if strategy == "linear":
        param_groups = [{'params': (p for n, p in encoder.named_parameters() if 'head' in n)}]
    elif strategy == "full":
        param_groups = [
            {
                'params': (p for n, p in encoder.named_parameters() if ('bias' not in n) and (len(p.shape) != 1)),
            }, {
                'params': (p for n, p in encoder.named_parameters() if ('bias' in n) or (len(p.shape) == 1)),
                'weight_decay': 0.,
                'WD_exclude': True,
            }
        ]
        
    # Optimizer for downstream task
    logger.info('Using AdamW')
    optimizer = torch.optim.AdamW(param_groups)
    if scheduler_type == "cosine":
        scheduler = WarmupCosineSchedule(
            optimizer,
            warmup_steps=int(warmup * iterations_per_epoch),
            start_lr=start_lr,
            ref_lr=ref_lr,
            final_lr=final_lr,
            T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
        )
    elif scheduler_type == "linear":
        scheduler = LinearDecaySchedule(
            optimizer,
            start_lr=start_lr,
            final_lr=final_lr,
            T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
        )
    else:
        raise ValueError(f"Scheduler type {scheduler_type} not recognized")
    wd_scheduler = CosineWDSchedule(
        optimizer=optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_bfloat16) if use_bfloat16 else None
    return optimizer, scaler, scheduler, wd_scheduler, encoder