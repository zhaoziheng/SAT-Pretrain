import torch
import torch.nn.functional as F
from einops import repeat

def collate_atlas(data):
    """
    Pad 3D images to the same depth
    
    Args:
        data : [{'label_name':..., 'id_ls':..., 'image':..., 'mask':..., 'modality':..., 'image_path':..., 'mask_path':..., 'y1x1z1_y2x2z2':...}, ...]
    """
    
    image = []
    mask = []
    label_name = []
    id_ls = []
    modality = []
    image_path = []
    mask_path = []
    y1x1z1_y2x2z2 = []
    
    # pad to max num of class in the batch
    max_class = 10
    query_mask = torch.zeros((len(data), max_class), dtype=torch.bool) # b n
    for i, sample in enumerate(data):
        if sample['image'].shape[0] == 1:
            sample['image'] = repeat(sample['image'], 'c h w d -> (c r) h w d', r=3)
        image.append(sample['image'])
        
        class_num = sample['mask'].shape[0]
        pad = (0, 0, 0, 0, 0, 0, 0, max_class-class_num)
        padded_mask = F.pad(sample['mask'], pad, 'constant', 0)   # nhwd
        mask.append(padded_mask)
        
        sample['label_name'] += ['none'] * (max_class-class_num)
        sample['id_ls'] += ['none'] * (max_class-class_num)
        query_mask[i, :class_num] = True
        label_name.append(sample['label_name']) # [[l1, l2, ...], ..., 'none']
        id_ls.append(sample['id_ls']) # [[id1, id2, ...], ..., 'none']
        
        modality.append(sample['modality']) # [0, 1, ...]
        image_path.append(sample['image_path'])
        mask_path.append(sample['mask_path'])
        y1x1z1_y2x2z2.append(sample['y1x1z1_y2x2z2'])
        
    image = torch.stack(image, dim=0).float()   # bchwd
    mask = torch.stack(mask, dim=0).float() # bnhwd
    return {'image':image, 'mask':mask, 'label_name':label_name, 'id_ls':id_ls, 'modality':modality, 'image_path':image_path, 'mask_path':mask_path, 'query_mask':query_mask, 'y1x1z1_y2x2z2':y1x1z1_y2x2z2}


def collate_text(data):
    label_text = []
    pos_text = []
    
    for sample in data:
        
        if isinstance(sample['label_text'], list):
            label_text += sample['label_text']
            pos_text += sample['pos_text']
            
        else:
            label_text.append(sample['label_text'])
            pos_text.append(sample['pos_text'])
            
    return {'label_text':label_text, 'pos_text':pos_text}