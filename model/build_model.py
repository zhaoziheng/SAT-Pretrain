import torch
import torch.nn as nn
import os

from .knowledge_encoder import Knowledge_Encoder
from .text_tower import Text_Tower
from .atlas_tower import Atlas_Tower

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def build_model(args, device, gpu_id):
    text_tower = Text_Tower(
        args.medcpt_checkpoint, 
        args.pubmedbert_checkpoint, 
        args.biolord_checkpoint,
        args.out_dim, 
        args.open_bert_layer,
        args.max_text_length,
        args.random_truncate_prob)
    
    atlas_tower = Atlas_Tower(
        out_dim=args.out_dim,
        vision_backbone=args.vision_backbone
    )
    
    model = Knowledge_Encoder(
        atlas_tower=atlas_tower,
        text_tower=text_tower,
        # use_pretrained_text_tower=args.use_pretrained_text_tower,
        # text_tower_checkpoint=args.text_tower_checkpoint,
        # use_pretrained_unet_encoder=args.use_pretrained_unet_encoder,
        # unet_encoder_checkpoint=args.unet_encoder_checkpoint
    )
    
    if args.pretrain_text_tower:
        model.lock_atlas_tower()
    
    # multi-gpu parallel
    if "RANK" in os.environ:
        model = model.to(device)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)        
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu_id], find_unused_parameters=True)
    else:
        device = torch.device('cuda')
        model = nn.DataParallel(model)
        model.to(device)
        
    if "RANK" not in os.environ or int(os.environ["RANK"]) == 0:
        print(f"** MODEL ** {get_parameter_number(model)['Total']/1e6}M parameters")
        print(f"** TEXT TOWER ** {get_parameter_number(text_tower)['Total']/1e6}M parameters")
        print(f"** ATLAS TOWER ** {get_parameter_number(atlas_tower)['Total']/1e6}M parameters")
        
    return model


# v2, load text tower and modality embedding
def build_text_encoder(args):
    text_tower = Text_Tower(
        args.medcpt_checkpoint, 
        args.pubmedbert_checkpoint, 
        args.biolord_checkpoint,
        args.out_dim, 
        args.open_bert_layer,
        args.max_text_length,
        args.random_truncate_prob)
    
    atlas_tower = None
    
    model = Knowledge_Encoder(
        atlas_tower=atlas_tower,
        text_tower=text_tower,
        # use_pretrained_text_tower=False,
        # text_tower_checkpoint=None,
        # use_pretrained_unet_encoder=False,
        # unet_encoder_checkpoint=None
        )
        
    return model


def load_checkpoint(checkpoint, 
                    resume, 
                    partial_load, 
                    model, 
                    optimizer, 
                    device):
    
    if int(os.environ["RANK"]) == 0:
        print('** CHECKPOINT ** : Load checkpoint from %s' % (checkpoint))
    
    if "RANK" in os.environ:
        checkpoint = torch.load(checkpoint, map_location=device)
    else:
        checkpoint = torch.load(checkpoint, map_location='cpu')
        
    # load part of the checkpoint
    if partial_load:
        model_dict =  model.state_dict()
        # check difference
        unexpected_state_dict = [k for k in checkpoint['model_state_dict'].keys() if k not in model_dict.keys()]
        missing_state_dict = [k for k in model_dict.keys() if k not in checkpoint['model_state_dict'].keys()]
        unmatchd_state_dict = [k for k,v in checkpoint['model_state_dict'].items() if k in model_dict.keys() and v.shape != model_dict[k].shape]
        # load partial parameters
        state_dict = {k:v for k,v in checkpoint['model_state_dict'].items() if k in model_dict.keys() and v.shape == model_dict[k].shape}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
        if int(os.environ["RANK"]) == 0:
            print('The following parameters are unexpected in SAT checkpoint:\n', unexpected_state_dict)
            print('The following parameters are missing in SAT checkpoint:\n', missing_state_dict)
            print('The following parameters have different shapes in SAT checkpoint:\n', unmatchd_state_dict)
            print('The following parameters are loaded in SAT:\n', state_dict.keys())
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
        
    # if resume, load optimizer and step
    if resume:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except:
            pass
        start_step = int(checkpoint['step']) + 1
    else:
        start_step = 1
        
    return model, optimizer, start_step