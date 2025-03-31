import os

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .dataset import Uni_Mod_Dataset, Med_SAM_Dataset, Med_SAM_Dataset_npz
from .collate import collate_atlas, collate_text


def build_dataset(args):
    
    text_set = Uni_Mod_Dataset(
            umls_def_file = args.umls_def_file,
            umls_kg_file = args.umls_kg_file,
            website_knowledge_file = args.website_knowledge_text_file,
            supplementary_file = args.supplementary_text_file,
            sample_umls_def_ratio = args.sample_umls_def_ratio, 
            sample_umls_kg_ratio = args.sample_umls_kg_ratio, 
            sample_website_knowledge_def_ratio = args.sample_website_knowledge_def_ratio,
            sample_website_knowledge_kg_ratio = args.sample_website_knowledge_kg_ratio,
            sample_supplementary_def_ratio = args.sample_supplementary_def_ratio,
            sample_supplementary_kg_ratio = args.sample_supplementary_kg_ratio,
            hard_negative_prob = args.hard_negative_prob
        )
    text_sampler = DistributedSampler(text_set)
    if args.num_workers is not None:
        text_loader = DataLoader(text_set, sampler=text_sampler, batch_size=args.batchsize_text, pin_memory=args.pin_memory, num_workers=args.num_workers, collate_fn=collate_text)
    else:
        text_loader = DataLoader(text_set, sampler=text_sampler, batch_size=args.batchsize_text, pin_memory=args.pin_memory, collate_fn=collate_text)
    
    if args.pretrain_text_tower:
        atlas_set = atlas_loader = atlas_sampler = None
    else:
        if args.data_source == 'cvpr25':
            atlas_set = Med_SAM_Dataset_npz(
                jsonl_file=args.sat_ds_data_jsonl,
                crop_size=[288,288,96],
                # uncenter_prob=args.uncenter_prob,
                # eval_mode=False
            )
        elif args.data_source == 'sat-ds':
            atlas_set = Med_SAM_Dataset(
                jsonl_file=args.sat_ds_data_jsonl,
                crop_size=[288,288,96],
                # uncenter_prob=args.uncenter_prob,
                # eval_mode=False
            )
        else:
            raise ValueError(f'Unknown data source: {args.data_source}')
        atlas_sampler = DistributedSampler(atlas_set)
        if args.num_workers is not None:
            atlas_loader = DataLoader(atlas_set, sampler=atlas_sampler, batch_size=args.batchsize_3d, pin_memory=args.pin_memory, num_workers=args.num_workers, collate_fn=collate_atlas)
        else:
            atlas_loader = DataLoader(atlas_set, sampler=atlas_sampler, batch_size=args.batchsize_3d, pin_memory=args.pin_memory, collate_fn=collate_atlas)
    
    return text_set, text_loader, text_sampler, atlas_set, atlas_loader, atlas_sampler
        