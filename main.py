import os
import random
from datetime import timedelta

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import optim
import torch.distributed as dist

from data.build_dataset import build_dataset

from model.build_model import build_model, load_checkpoint

from train.dist import is_master
from train.params import parse_args
from train.logger import set_up_log
from train.loss import ClipLoss_no_gather, ClipLoss
from train.scheduler import cosine_lr
from train.trainer import Trainer_LP, Trainer_VLP

def set_seed(config):
    seed = config.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    # new seed
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def main():
    # get configs
    args = parse_args()
    
    # set logger
    if not is_master():
        checkpoint_dir = None
        tb_writer = None
        log_file = None
    else:
        checkpoint_dir, tb_writer, log_file = set_up_log(args)
    
    # set random seed for reproducibility
    # set_seed(args)
    
    # set up distribution
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    gpu_id = int(os.environ["LOCAL_RANK"])
    torch.distributed.init_process_group(backend="nccl", init_method='env://', timeout=timedelta(minutes=60))
    # dispaly
    if is_master():
        print('** GPU NUM ** : ', torch.cuda.device_count())  # 打印gpu数量
        print('** WORLD SIZE ** : ', torch.distributed.get_world_size())
    rank = dist.get_rank()
    print(f"** DDP ** : Start running DDP on rank {rank}.")
    
    # dataset and loader
    text_set, text_loader, _, atlas_set, atlas_loader, _ = build_dataset(args)  
    
    # set model
    model = build_model(args, device, gpu_id)
    
    # set optimizer 
    exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "norm" in n or "bias" in n or 'temperature' in n
    include = lambda n, p: not exclude(n, p)
    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]
    optimizer = optim.AdamW(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.},
            {"params": rest_params, "weight_decay": args.weight_decay},
        ],
        lr=args.lr[0],
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )
    
    # set scheduler
    total_steps = args.step_num
    scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
    
    # if restart cosine annealing, total_steps = sum of steps in each stage
    if isinstance(total_steps, list):
        total_steps = sum(total_steps)
    
    # load checkpoint if specified
    start_step = 1
    if args.checkpoint:
        model, optimizer, start_step = load_checkpoint(
            checkpoint=args.checkpoint,
            resume=args.resume,
            partial_load=args.partial_load,
            model=model, 
            optimizer=optimizer, 
            device=device
        )
    if is_master():
        print(f'Starting from step {start_step}')
        
    if is_master():
        print('The following parameters in Knowledge Encoder are frozen:')
        for name, param in model.named_parameters():
            if not param.requires_grad:
                print(name)
        print('The following parameters in Knowledge Encoder are trainable:')
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)
    
    # set train function
    if args.pretrain_text_tower:
        trainer = Trainer_LP(
            model=model, 
            device=device, 
            dataset=text_set, 
            loader=text_loader, 
            loss=ClipLoss_no_gather(), 
            optimizer=optimizer, 
            scheduler=scheduler, 
            tb_writer=tb_writer, 
            checkpoint_dir=checkpoint_dir, 
            log_file=log_file, 
            args=args
        )
    else:
        trainer = Trainer_VLP(
            model=model, 
            device=device, 
            text_set=text_set, 
            text_loader=text_loader, 
            atlas_set=atlas_set, 
            atlas_loader=atlas_loader, 
            atlas_loss=ClipLoss(),
            text_loss=ClipLoss_no_gather(),
            optimizer=optimizer,
            scheduler=scheduler, 
            tb_writer=tb_writer, 
            checkpoint_dir=checkpoint_dir, 
            log_file=log_file, 
            args=args
            )
    
    for step in range(start_step, total_steps):
        
        # make sure the train is not interrupted
        if is_master() and step%100==0:
            print(f'Training Step %d'%step)
            
        # train the model in an epoch
        trainer.train_one_step(step)

if __name__ == '__main__':
    main()
    
    
    
        
    
    