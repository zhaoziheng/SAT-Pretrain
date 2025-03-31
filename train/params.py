'''
Parameters assignment
'''

import argparse

def str2bool(v):
    return v.lower() in ('true')

"""
def get_default_params(model_name):
    # Params from paper [CLIP](https://arxiv.org/pdf/2103.00020.pdf)
    model_name = model_name.lower()
    if "vit" in model_name:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    else:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}
"""

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Exp Controller
    
    # name and dir
    parser.add_argument(
        "--log_dir",
        type=str,
        default='/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/medical-knowledge-pretraining/log',
        help="Dir of exp logs",
    )
    parser.add_argument(
        "--rcd_dir",
        type=str,
        default=None,
        help="refs to evaluate.py, save the retrieval results",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Name of the exp",
    )
    
    # load checkpoint
    parser.add_argument(
        "--resume",
        type=str2bool,
        default=False,
        help="Resume an interrupted exp",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path",
    )
    parser.add_argument(
        "--partial_load",
        type=str2bool,
        default=False,
        help="Allow to load partial paramters from checkpoint",
    )
    
    # save and log
    parser.add_argument(
        "--save_large_interval",
        type=int, 
        default=1000,
        help="Save checkpoint every N steps as step_xxx.pth"
    )
    parser.add_argument(
        "--save_small_interval",
        type=int, 
        default=100,
        help="Save checkpoint every N steps as latest_step.pth"
    )
    parser.add_argument(
        "--eval_step_interval",
        type=int, 
        help="Evaluate every N steps"
    )
    
    # others
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed",
    )
    
    # loss, optimizer
    parser.add_argument(
        "--max_logit_scale",
        type=int,
        default=100,
        help="Clamp the logits scale for stability in training, exp(t) < max_logit_scale, t < ln(max_logit_scale)",
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.999
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1.0e-8
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
    )
    
    # Learing Rate 
    
    parser.add_argument(
        "--step_num",
        type=int,
        nargs='+',
        help="Total epoch num, can be multistage",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        nargs='+',
        help="Warm up step num, can be multistage",
    )
    parser.add_argument(
        "--lr",
        type=float,
        nargs='+',
        help="Peak learning rate, can be multistage",
    )
    
    # Text Dataset
    parser.add_argument(
        "--umls_def_file",
        type=str,
        default='/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SAT-Pretrain/data/knowledge_data/MRDEF_name.csv',
    )
    parser.add_argument(
        "--umls_kg_file",
        type=str,
        default='/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SAT-Pretrain/data/knowledge_data/umls_kg.csv',
    )
    parser.add_argument(
        "--website_knowledge_text_file",
        type=str,
        default='/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SAT-Pretrain/data/knowledge_data/website_knowledge_lab2def_gpt.json'
    )
    parser.add_argument(
        "--supplementary_text_file",
        type=str,
        default='/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SAT-Pretrain/data/knowledge_data/supplementary_lab2def_gpt.json'
    )
    # Ratio to control data source
    parser.add_argument(
        "--sample_umls_def_ratio",
        type=float,
        default=0.25,
        help='determine the num of UMLS Definition data in an epoch (when training Text-Tower)'
    )
    parser.add_argument(
        "--sample_umls_kg_ratio",
        type=float,
        default=0.05,
        help='determine the num of UMLS Definition data in an epoch (when training Text-Tower)'
    )
    parser.add_argument(
        "--sample_website_knowledge_def_ratio",
        type=float,
        default=1.0
    )
    parser.add_argument(
        "--sample_website_knowledge_kg_ratio",
        type=float,
        default=1.0
    )
    parser.add_argument(
        "--sample_supplementary_def_ratio",
        type=float,
        default=1.0
    )
    parser.add_argument(
        "--sample_supplementary_kg_ratio",
        type=float,
        default=1.0
    )
    parser.add_argument(
        "--hard_negative_prob",
        type=float,
        default=0.0
    )
    
    # Atlas Dataset
    parser.add_argument(
        "--sat_ds_data_jsonl",
        type=str,
        default='/mnt/hwfile/medai/zhaoziheng/SAM/trainsets_v4/merge/sat_ds_train_49_datasets.jsonl'
    )
    parser.add_argument(
        "--data_source",
        type=str,
        default='sat-ds'
    )
    
    # dataset, sampler and loader
    parser.add_argument(
        "--crop_size",
        type=int,
        nargs='+',
        default=[288, 288, 96]
    )
    parser.add_argument(
        "--batchsize_3d",
        type=int,
        default=16,
        help='size for a 3d atlas batch'
    )
    parser.add_argument(
        "--batchsize_text",
        type=int,
        default=256,
        help='batch size when pretraining text tower'
    )
    parser.add_argument(
        "--pin_memory",
        type=str2bool,
        default=True,
        help='load data to gpu to accelerate'
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8
    )
    
    # Atlas Tower Configs
    parser.add_argument(
        "--vision_backbone",
        type=str
    )
    parser.add_argument(
        "--out_dim",
        type=int,
        default=768,
        help='(Project atlas output feature to align with text feature) Should be consistent with Text Tower output dim'
    )
    
    # Text Tower Configs
    parser.add_argument(
        "--pretrain_text_tower",
        type=str2bool,
        help='Pretrained text tower via text-text CL, otherwise train text tower & altas tower simultaneously'
    )
    parser.add_argument(
        "--medcpt_checkpoint",
        type=str
    )
    parser.add_argument(
        "--pubmedbert_checkpoint",
        type=str
    )
    parser.add_argument(
        "--biolord_checkpoint",
        type=str,
        default='/mnt/hwfile/medai/zhaoziheng/SAM/Knowledge_Data/BioLORD2023C'
    )
    parser.add_argument(
        "--open_bert_layer",
        type=int,
        help='from -1(totally open except for word embedding) to 11(freeze bert, only open projection layer)'
    )
    
    # Tokenizer
    parser.add_argument(
        "--max_text_length",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--random_truncate_prob",
        type=float,
        default=0.5,
    )
    
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    import torch
    
    a = torch.load('../others/xiaoman_bert/epoch_latest.pt')
    print('bert_model.embeddings.position_ids' in a['state_dict'])