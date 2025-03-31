# (Optional) Stage 1 Textual Knowledge Pretraining

torchrun \
--nnodes 1 \
--nproc_per_node 1 \
--master_port 25996 \
/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SAT-Pretrain/main.py \
--name 'textual_pretraining' \
--log_dir '/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SAT-Pretrain/log' \
--vision_backbone 'UNET' \
--resume False \
--partial_load True \
--save_large_interval 10000 \
--save_small_interval 2000 \
--eval_step_interval 2000 \
--step_num 100000 \
--warmup 10000 \
--lr 1e-5 \
--website_knowledge_text_file '/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SAT-Pretrain/data/knowledge_data/website_knowledge_lab2def_gpt.json' \
--supplementary_text_file '/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SAT-Pretrain/data/knowledge_data/supplementary_lab2def_gpt.json' \
--umls_def_file '/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SAT-Pretrain/data/knowledge_data/umls_def.csv' \
--umls_kg_file '/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SAT-Pretrain/data/knowledge_data/umls_kg.csv' \
--sample_umls_def_ratio 0.1 \
--sample_umls_kg_ratio 0.02 \
--sample_website_knowledge_def_ratio 1.0 \
--sample_website_knowledge_kg_ratio 0.5 \
--biolord_checkpoint '/mnt/hwfile/medai/zhaoziheng/SAM/Knowledge_Data/BioLORD2023C' \
--open_bert_layer 8 \
--batchsize_text 128 \
--max_text_length 256 \
--random_truncate_prob 0.5 \
--pretrain_text_tower 'True' \
--num_workers 16 \
--pin_memory False

# Stage 2 Multimodel Pretraining

torchrun \
--nnodes 1 \
--nproc_per_node 2 \
--master_port 25996 \
/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SAT-Pretrain/main.py \
--name 'multimodal_pretraining' \
--log_dir '/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SAT-Pretrain/log' \
--vision_backbone 'UNET' \
--checkpoint '/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SAT-Pretrain/log/textual_pretraining/checkpoint/latest_step.pth' \
--resume False \
--partial_load True \
--save_large_interval 10000 \
--save_small_interval 2000 \
--eval_step_interval 100 \
--step_num 100000 \
--warmup 10000 \
--lr 1e-5 \
--data_source 'cvpr25' \
--sat_ds_data_jsonl '/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SAT-Pretrain/data/knowledge_data/train_10percent.jsonl' \
--batchsize_3d 2 \
--website_knowledge_text_file '/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SAT-Pretrain/data/knowledge_data/website_knowledge_lab2def_gpt.json' \
--supplementary_text_file '/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SAT-Pretrain/data/knowledge_data/supplementary_lab2def_gpt.json' \
--umls_def_file '/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SAT-Pretrain/data/knowledge_data/umls_def.csv' \
--umls_kg_file '/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SAT-Pretrain/data/knowledge_data/umls_kg.csv' \
--sample_umls_def_ratio 0.1 \
--sample_umls_kg_ratio 0.02 \
--sample_website_knowledge_def_ratio 1.0 \
--sample_website_knowledge_kg_ratio 0.5 \
--biolord_checkpoint '/mnt/hwfile/medai/zhaoziheng/SAM/Knowledge_Data/BioLORD2023C' \
--open_bert_layer 8 \
--batchsize_text 128 \
--max_text_length 256 \
--random_truncate_prob 0.5 \
--pretrain_text_tower 'False' \
--num_workers 16 \
--pin_memory False