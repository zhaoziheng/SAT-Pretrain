import torch
import torch.nn.functional as F
import numpy as np

def get_labels(text_lists):
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
    # labels = F.normalize(labels,dim=0)
    return labels

def get_retrieval_metrics(feature1, feature2, type1='input_1', type2='input_2', logit_scale=1.0, text_list_or_gt_matrix=None):
    """
    retrieval metric
    
    Args:
        feature1/2 (tensor or array or path to npy file): shape (N, dim)
        type1/2 (str): sample type. e.g. image, atlas, text, label_name, definition ...
        logit_scale (float): exp(t)

    Returns:
        avg and median of gt ranks, and recall rate;
        top10 predicts of each sample;
        gt ranks of each sample;
    """
    if isinstance(feature1, str):
        feature1 = torch.tensor(np.load(feature1), dtype=torch.float32) 
    if isinstance(feature2, str):
        feature2 = torch.tensor(np.load(feature2), dtype=torch.float32)
    
    if isinstance(feature1, np.ndarray):
        feature1 = torch.tensor(feature1, dtype=torch.float32).detach().cpu()
    if isinstance(feature2, np.ndarray):
        feature2 = torch.tensor(feature2, dtype=torch.float32).detach().cpu()
    
    # features: [data_size, dim]
    feature1 = F.normalize(feature1, dim=-1)
    feature2 = F.normalize(feature2, dim=-1)
    
    metrics = {}
    logits_per_sample1 = (logit_scale * feature1 @ feature2.t()) # [n1, n2]
    logits_per_sample2 = logits_per_sample1.t()

    logits = {f"{type1}_to_{type2}": logits_per_sample1, f"{type2}_to_{type1}": logits_per_sample2}

    # get gt matrix : 0,1 matrix of shape (n, n)
    if isinstance(text_list_or_gt_matrix, list):
        gt_matrix = get_labels(text_list_or_gt_matrix)
    elif isinstance(text_list_or_gt_matrix, torch.Tensor):
        gt_matrix = text_list_or_gt_matrix
    else:
        gt_matrix = torch.eye(feature1.shape[0])
        
    top10 = {}
    gt_ranks = {}
    for name, logit in logits.items():
        # a sample may have multiple gt, find the one with highest logits
        gt_logit = gt_matrix * logit  # [0.1, 0.3, 0.2] * [1, 0, 1] --> [0.1, 0, 0.2]
        gt_rank = torch.argmax(gt_logit, dim=1).view(-1, 1)  # --> 2

        ranking = torch.argsort(logit, descending=True) # [0.1, 0.3, 0.2] --> [1, 2, 0] ranking logits along each row
        preds = torch.where(ranking == gt_rank)[1] # [1, 2, 0] == [0] --> [2] where the gt lies in ranks
        preds = preds.detach().cpu().numpy()
        
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)
        top10[f"{name}_top10_pred_idx"] = ranking[:, 0:10] # [n, 10]
        gt_ranks[f"{name}_gt_rank"] = preds    # [n]

    logits_num = feature1.shape[0]
    return logits_num, metrics, top10, gt_ranks

if __name__ == '__main__':
    a = torch.rand((4, 5))
    b = torch.rand((4, 5))
    text_list = ['1', '2', '1', '1']
    logits_num, metrics, top10_pred_idx, gt_rank = get_retrieval_metrics(a, b, text_list=text_list)
    print(metrics)