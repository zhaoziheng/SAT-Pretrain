import os

import torch
import torch.distributed as dist
from torch.distributed import nn as dist_nn
from torch.nn import functional as F
import torch.nn as nn
import numpy as np


def multi_positive_infoNCE_loss(pred_sim, true_sim, positive_threshold=0.7):
    """
    infoNCE改进版本，分母去掉positive的元素（除了对角线）
    """
    true_sim[true_sim>=positive_threshold] = 1
    true_sim[true_sim<positive_threshold] = 0
    
    diag_mask = torch.eye(true_sim.shape[0]).to(true_sim.device)
    undiag_mask = torch.zeros_like(true_sim)
    undiag_mask[true_sim==0] = 1
    mask = undiag_mask + diag_mask

    logits_sum = torch.exp(pred_sim).mul(mask).sum(1)
    logits_norm = torch.exp(torch.diag(pred_sim)) / logits_sum
    loss = -1 * torch.log(logits_norm)
    loss = loss.mean()
    
    return loss


def gather_features(
        image_features,
        text_features,
        id_list=None,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
):
        
    if gather_with_grad:
        all_image_features = torch.cat(dist_nn.functional.all_gather(image_features), dim=0)    # N --> N * Word_Size
        all_text_features = torch.cat(dist_nn.functional.all_gather(text_features), dim=0)
    else:
        gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
        gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)  # no grad at all
        if not local_loss:
            # ensure grads for local rank when all_* features don't have a gradient
            gathered_image_features[rank] = image_features
            gathered_text_features[rank] = text_features
        all_image_features = torch.cat(gathered_image_features, dim=0)
        all_text_features = torch.cat(gathered_text_features, dim=0)
    
    all_id_list = [None for _ in range(world_size)]
    dist.all_gather_object(all_id_list, id_list)
    all_id_list = [id for id_ls in all_id_list for id in id_ls]
    
    return all_image_features, all_text_features, all_id_list


class ClipLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.local_loss = False
        self.gather_with_grad = True
        
    def get_gt_matrix(self, text_lists):
        #caculate labels (find repeated ones)
        num_logits = len(text_lists)
        labels = torch.eye(num_logits, dtype=torch.float)
        for i in range(num_logits):
            search_keywords = text_lists[i]
            for j in range(i,num_logits):
                match_keywords = text_lists[j]
                if search_keywords == match_keywords:
                    labels[i,j]=1
                    labels[j,i]=1
        return labels   # matrix with multi positive elements per row/column

        # /mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/medical-knowledge-pretraining/src/train
    
    def get_logits_and_gt(self, image_features, text_features, ids, logit_scale, device):
        # get logits
        if "RANK" in os.environ and dist.get_world_size() > 1:
            all_image_features, all_text_features, all_ids = gather_features(
                image_features, text_features, ids,
                self.local_loss, self.gather_with_grad, dist.get_rank(), dist.get_world_size())
            
            # print(f'rank {dist.get_rank()}, local {image_features.shape}, {len(ids)}, global {all_image_features.shape}, {len(all_ids)}')
            
            # filter padding in gathered features after gather
            query_mask = [0] * all_image_features.shape[0]
            tmp = []
            for i, lab_mod in enumerate(all_ids):
                if lab_mod != 'none':
                    query_mask[i] = 1
                    tmp.append(lab_mod)
            query_mask = torch.tensor(query_mask, dtype=torch.bool)
            all_ids = tmp
            all_text_features = all_text_features[query_mask]
            all_image_features = all_image_features[query_mask]
            
            # filter padding in local features after gather
            query_mask = [0] * image_features.shape[0]
            for i, lab_mod in enumerate(ids):
                if lab_mod != 'none':
                    query_mask[i] = 1
            text_features = text_features[query_mask]
            image_features = image_features[query_mask]

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T # local_n * N
                logits_per_text = logit_scale * text_features @ all_image_features.T
                
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T # N * N
                logits_per_text = logits_per_image.T

        else:
            # filter padding in local features after gather
            query_mask = [0] * image_features.shape[0]
            tmp = []
            for i, lab_mod in enumerate(ids):
                if lab_mod != 'none':
                    query_mask[i] = 1
                    tmp.append(lab_mod)
            all_ids = tmp
            all_text_features = text_features[query_mask]
            all_image_features = image_features[query_mask]
            
            logits_per_image = logit_scale * all_image_features @ all_text_features.T # N * N
            logits_per_text = logit_scale * all_text_features @ all_image_features.T
            
        local_logits = logits_per_image.shape[0] # local_n / N * N
        
        # get gt matrix
        gt_matrix = self.get_gt_matrix(all_ids)
        gt_matrix = gt_matrix.to(device)
        gt_matrix = gt_matrix[:local_logits, :] # local_n / N * N
        
        return logits_per_image, logits_per_text, gt_matrix

    def forward(self, prediction, id_list):
        image_features = prediction['image_features']
        text_features = prediction['text_features']
        logit_scale = prediction['logit_scale'] # NOTE: logits_scale = exp(t)
        
        # NOTE: the features should be normalized before
        device = image_features.device  
        logits_per_image, logits_per_text, gt = self.get_logits_and_gt(image_features, text_features, id_list, logit_scale, device)

        # total_loss = (
        #     F.cross_entropy(logits_per_image, gt) +
        #     F.cross_entropy(logits_per_text, gt)
        #     ) / 2
        
        total_loss = (
            multi_positive_infoNCE_loss(logits_per_image, gt) +
            multi_positive_infoNCE_loss(logits_per_text, gt)
            ) / 2
        

        return total_loss


class ClipLoss_no_gather(nn.Module):
    def __init__(self, moco_size=0):
        super().__init__()
        
    def get_gt_matrix(self, id_lists):
        #caculate labels (find repeated ones)
        num_logits = len(id_lists)
        labels = torch.eye(num_logits, dtype=torch.float)
        for i in range(num_logits):
            search_keywords = id_lists[i]
            for j in range(i,num_logits):
                match_keywords = id_lists[j]
                if search_keywords == match_keywords:
                    labels[i,j]=1
                    labels[j,i]=1
        labels = F.normalize(labels,dim=0)
        return labels
    
    def get_logits_and_gt(self, image_features, text_features, ids, logit_scale, device):
        # get gt  
        gt_matrix = self.get_gt_matrix(ids)
        gt_matrix = gt_matrix.to(device)
        
        # no gather, just a single gpu            
        logits_per_image = logit_scale * image_features @ text_features.T # N * N
        logits_per_text = logit_scale * text_features @ image_features.T

        
        return logits_per_image, logits_per_text, gt_matrix

    def forward(self, prediction, id_list=None):
        image_features = prediction['image_features']
        text_features = prediction['text_features']
        logit_scale = prediction['logit_scale'] # NOTE: logits_scale = exp(t)
        
        # NOTE: the features should be normalized before
        device = image_features.device  
        logits_per_image, logits_per_text, gt = self.get_logits_and_gt(image_features, text_features, id_list, logit_scale, device)

        total_loss = (
            F.cross_entropy(logits_per_image, gt) +
            F.cross_entropy(logits_per_text, gt)
            ) / 2

        return total_loss

    