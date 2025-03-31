import math
import os
import time

from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
from einops import rearrange
import torch

from .metric import get_retrieval_metrics
from .dist import is_master

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.val = 0

    def update(self, val, n=1): # avg value over samples, and the num of samples, e.g. the avg loss over a batch and batchsize
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

class Trainer_VLP():
    def __init__(
        self,
        model, 
        device, 
        text_set, 
        text_loader,
        atlas_set, 
        atlas_loader, 
        atlas_loss,
        text_loss,
        optimizer,
        scheduler, 
        tb_writer, 
        checkpoint_dir, 
        log_file, 
        args
        ):
        
        # model
        self.model = model
        self.device = device
        self.max_logit_scale = args.max_logit_scale # clamp ttttt

        # data
        self.text_set = text_set
        self.text_loader = iter(text_loader)
        self.atlas_set = atlas_set
        self.atlas_loader = iter(atlas_loader)
        
        # loss calculator, optimizer and lr
        self.atlas_loss = atlas_loss
        self.text_loss = text_loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        # amp
        self.atlas_grad_scaler = GradScaler()
        self.text_grad_scaler = GradScaler()
        
        # logger
        self.tb_writer = tb_writer
        self.checkpoint_dir = checkpoint_dir
        self.log_file = log_file
        
        self.text_loss_m = AverageMeter()
        self.atlas_loss_m = AverageMeter()
        self.text_time_m = AverageMeter()
        self.atlas_data_time_m = AverageMeter()
        self.atlas_fwd_time_m = AverageMeter()
        self.atlas_bwd_time_m = AverageMeter()
        
        # step interval
        self.save_small_interval = args.save_small_interval
        self.save_large_interval = args.save_large_interval
        self.eval_step_interval = args.eval_step_interval
        
        # dynamic evaluation
        self.atlas_text_banks = {'atlas_features':[], 'text_features':[], 'labels':[]}
        self.text_text_banks = {'text1_features':[], 'text2_features':[], 'proj_text1_features':[], 'proj_text2_features':[], 'labels':[]}
        
        
    def train_one_step(self, step):
        self.model.train()
        self.scheduler(step)
        
        #######################
        # Text - Atlas Contrast
        #######################
        end_time = time.time()
        
        atlas_batch = next(self.atlas_loader)
        input_text = atlas_batch['label_name']  # [[l1, l2, ...], ..., 'none'] b n
        id_list = atlas_batch['id_ls'] # [[id1, id2, ...], ..., 'none'] b n
        input_image = atlas_batch['image'] # bchwd
        input_mask = atlas_batch['mask'] # bnhwd
        modality_code = atlas_batch['modality'] # [m1, ...] b
        query_mask = atlas_batch['query_mask']  # b n
        max_class_num = len(input_text[0])
        
        # unfold text and align with modality
        input_text = [l for ls in input_text for l in ls]   # b n -> bn
        id_list = [i for ls in id_list for i in ls]   # bn
        modality_code = [m for m in modality_code for i in range(max_class_num)] # bn
        modality_code = torch.tensor(modality_code)
        
        # # unique embedding = label + modality
        # id_list = []
        # for lbl_name, mod_code in zip(input_text, modality_code):
        #     if lbl_name != 'none':
        #         id_list.append(f'{lbl_name} - {mod_code}')
        #     else:
        #         id_list.append('none')
                                   
        input_image = input_image.to(device=self.device)
        input_mask = input_mask.to(device=self.device)
            
        self.atlas_data_time_m.update(time.time()-end_time)
        end_time = time.time()
            
        # forward and backward
        self.optimizer.zero_grad()
            
        with autocast():
            text_feature, altas_feature, _ = self.model(
                text1=input_text, 
                text2=None, 
                image=input_image, 
                mask=input_mask, 
                modality=modality_code
                )   # b n d, (bn)d
            
            altas_feature = rearrange(altas_feature, 'b n d -> (b n) d')
            
            # filter padding label            
            query_mask = rearrange(query_mask, 'b n -> (b n)')
            unpadded_text_feature = text_feature[query_mask] # k d, k = True nums in query mask
            unpadded_altas_feature = altas_feature[query_mask]
            unpadded_id_list = [f'{id_name} - {mod_code}' for id_name, mod_code in zip(id_list, modality_code) if id_name!='none'] # k,  k = True nums in query mask
            
            # dynamic evaluation
            if is_master():
                self.atlas_text_banks['atlas_features'].append(unpadded_altas_feature.detach().cpu())
                self.atlas_text_banks['text_features'].append(unpadded_text_feature.detach().cpu())
                self.atlas_text_banks['labels'] += unpadded_id_list
                
            self.atlas_fwd_time_m.update(time.time()-end_time)
            end_time = time.time()

            # to gather features between devices, still need padding
            prediction = {
                'image_features':altas_feature,
                'text_features':text_feature,
                'logit_scale':self.model.module.temperature.exp()
                }   # 计算梯度需要保留logit_scale为张量
            total_loss = self.atlas_loss(prediction, id_list=id_list)   # avg within a batch

        self.atlas_grad_scaler.scale(total_loss).backward()
        self.atlas_grad_scaler.step(self.optimizer)
        self.atlas_grad_scaler.update()
        
        # Note: we clamp t to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            self.model.module.temperature.clamp_(0, math.log(self.max_logit_scale))
            
        batch_size = len(unpadded_id_list)   # bs = bs * n
        self.atlas_loss_m.update(total_loss.item(), batch_size)

        self.atlas_bwd_time_m.update(time.time()-end_time)
        end_time = time.time()
        
        # 清理显存占用
        torch.cuda.empty_cache()
            
        #######################
        # Text - Text Contrast
        #######################
        
        text_batch = next(self.text_loader)
            
        # data loading
        label_text, pos_text = text_batch['label_text'], text_batch['pos_text']
        
        # forward and backward
        self.optimizer.zero_grad()
        
        with autocast():
            text1_feature, text2_feature, proj_text1_feature, proj_text2_feature, _ = self.model(
                text1=label_text, 
                text2=pos_text, 
                image=None, 
                mask=None, 
                modality=None
                )
            
            if is_master():
                self.text_text_banks['text2_features'].append(text2_feature.detach().cpu())
                self.text_text_banks['text1_features'].append(text1_feature.detach().cpu())
                self.text_text_banks['labels'] += label_text
        
            prediction = {
                'image_features':text1_feature, 
                'text_features':text2_feature, 
                'logit_scale':self.model.module.temperature.exp()
                }  # 计算梯度需要保留logit_scale为张量
            total_loss = self.text_loss(prediction, id_list=label_text)   # avg within a batch
            
            prediction = {
                'image_features':proj_text1_feature, 
                'text_features':proj_text2_feature, 
                'logit_scale':self.model.module.temperature.exp()
                }  # 计算梯度需要保留logit_scale为张量
            total_loss += self.text_loss(prediction, id_list=label_text)   # avg within a batch
        
        self.text_grad_scaler.scale(total_loss).backward()
        self.text_grad_scaler.step(self.optimizer)
        self.text_grad_scaler.update()

        # NOTE: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            self.model.module.temperature.clamp_(0, math.log(self.max_logit_scale))  # exp(t) < 100, t < ln(100)
        
        batch_size = len(label_text)
        self.text_loss_m.update(total_loss.item(), batch_size)  

        # statistic time
        self.text_time_m.update(time.time()-end_time)
        end_time = time.time()
        
        # 清理显存占用
        torch.cuda.empty_cache()
                
        # log regularly (by epoch)
        with torch.no_grad():
            if is_master():
                if step % self.eval_step_interval == 0:
                    lr = self.optimizer.param_groups[0]['lr']
                    self.tb_writer.add_scalar('train/atlas_loss', self.atlas_loss_m.avg, step)
                    self.tb_writer.add_scalar('train/text_loss', self.text_loss_m.avg, step)
                    self.tb_writer.add_scalar('train/learning_rate', lr, step)
                    self.tb_writer.add_scalar('train/logit_scale', self.model.module.temperature.exp().item(), step)
                    info = f"Step {step} | LR {lr:.4e} | Logits Scale {self.model.module.temperature.exp().item():.4f}\n"
                    info += f"Atlas Loss {self.atlas_loss_m.val:.4f}({self.atlas_loss_m.avg:.4f}) | Text Loss {self.text_loss_m.val:.4f}({self.text_loss_m.avg:.4f})\n"
                    info += f"Atlas Data Time {self.atlas_data_time_m.avg:.2f} | Atlas Fwd Time {self.atlas_fwd_time_m.avg:.2f} | Atlas Bwd Time {self.atlas_bwd_time_m.avg:.2f} | Text Time {self.text_time_m.avg:.2f}\n"

                    # eval on dynamic feature bank (atlas-text)
                    feature_bank1 = torch.concat(self.atlas_text_banks['text_features'], dim=0)
                    self.atlas_text_banks['text_features'] = []
                    feature_bank2 = torch.concat(self.atlas_text_banks['atlas_features'], dim=0)    # n,d
                    self.atlas_text_banks['atlas_features'] = []
                    id_bank = self.atlas_text_banks['labels']
                    self.atlas_text_banks['labels'] = []
                    # cut if the matrix is tooo large
                    if len(id_bank) > 10000:
                        feature_bank1 = feature_bank1[len(id_bank)-10000:, :]
                        feature_bank2 = feature_bank2[len(id_bank)-10000:, :]
                        id_bank = id_bank[len(id_bank)-10000:]
                    info += f'Feature Bank Size : {feature_bank1.shape[0]} \n'
                    logits_num, metrics, top3_pred_idx, gt_rank = get_retrieval_metrics(
                        feature_bank1, feature_bank2, 
                        'text', 'atlas', 
                        self.model.module.temperature.exp().item(), 
                        text_list_or_gt_matrix=id_bank
                        )
                    for name, value in metrics.items():
                        if 'R@' in name:
                            info += f'| {name} {value:.4f} '
                            self.tb_writer.add_scalar(f'train/{name}', value, step)
                    info += '\n' 
                    
                    # eval on dynamic feature bank (text)
                    feature_bank1 = torch.concat(self.text_text_banks['text1_features'], dim=0)    # n,d
                    self.text_text_banks['text1_features'] = []
                    feature_bank2 = torch.concat(self.text_text_banks['text2_features'], dim=0)
                    self.text_text_banks['text2_features'] = []
                    id_bank = self.text_text_banks['labels']
                    self.text_text_banks['labels'] = []
                    # cut if the matrix is tooo large
                    if len(id_bank) > 10000:
                        feature_bank1 = feature_bank1[len(id_bank)-10000:, :]
                        feature_bank2 = feature_bank2[len(id_bank)-10000:, :]
                        id_bank = id_bank[len(id_bank)-10000:]
                    info += f'Feature Bank Size : {feature_bank1.shape[0]} \n'
                    logits_num, metrics, top3_pred_idx, gt_rank = get_retrieval_metrics(
                        feature_bank1, feature_bank2, 
                        'text1', 'text2', 
                        self.model.module.temperature.exp().item(), 
                        text_list_or_gt_matrix=id_bank
                        )
                    for name, value in metrics.items():
                        if 'R@' in name:
                            info += f'| {name} {value:.4f} '
                            self.tb_writer.add_scalar(f'train/{name}', value, step)
                    info += '\n'
                        
                    self.text_loss_m.reset()
                    self.atlas_loss_m.reset()
                    self.text_time_m.reset()
                    self.atlas_data_time_m.reset()
                    self.atlas_fwd_time_m.reset()
                    self.atlas_bwd_time_m.reset()
                                
                    # print log
                    print(info)
                    # write log
                    with open(self.log_file, 'a') as f:
                        f.write(info)
                
                # save checkpoint regularly (by epoch)
                if (step) % self.save_large_interval == 0:
                    torch.save({'step':step,
                                'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),         
                                'atlas_loss': self.atlas_loss_m.avg,
                                'text_loss': self.text_loss_m.avg
                                }, os.path.join(self.checkpoint_dir, f'step_{step}.pth'))
                    print(f'** CHECKPOINT ** step_{step}.pth')
                if (step) % self.save_small_interval == 0:
                    torch.save({'step':step,
                                'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),         
                                'atlas_loss': self.atlas_loss_m.avg,
                                'text_loss': self.text_loss_m.avg
                                }, os.path.join(self.checkpoint_dir, f'latest_step.pth'))
                    print(f'** CHECKPOINT ** latest_step.pth')
        
class Trainer_LP():
    def __init__(
        self,
        model, 
        device, 
        dataset, 
        loader, 
        loss, 
        optimizer, 
        scheduler, 
        tb_writer, 
        checkpoint_dir, 
        log_file, 
        args):
        
        # model
        self.model = model
        self.device = device
        self.max_logit_scale = args.max_logit_scale # clamp t

        # data
        self.dataset = dataset
        self.loader = iter(loader)
        
        # loss calculator, optimizer and lr
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        # amp
        self.grad_scaler = GradScaler()
        
        # logger
        self.tb_writer = tb_writer
        self.checkpoint_dir = checkpoint_dir
        self.log_file = log_file
        
        self.loss_m = AverageMeter()
        self.data_time_m = AverageMeter()
        self.fwd_time_m = AverageMeter()
        self.bwd_time_m = AverageMeter()
        
        # step interval
        self.save_small_interval = args.save_small_interval
        self.save_large_interval = args.save_large_interval
        self.eval_step_interval = args.eval_step_interval
        
        # dynamic evaluation
        self.text_text_banks = {'text1_features':[], 'text2_features':[], 'labels':[]}
        
    def train_one_step(self, step):
        self.model.train()
        self.scheduler(step)
            
        end_time = time.time()
        
        batch = next(self.loader)
        label_text, pos_text = batch['label_text'], batch['pos_text']
        
        # statistic time for loading data
        self.data_time_m.update(time.time()-end_time)
        end_time = time.time()
        
        # forward
        self.optimizer.zero_grad()
        
        with autocast():
            text1_feature, text2_feature, proj_text1_feature, proj_text2_feature, _ = self.model(
                text1=label_text, 
                text2=pos_text, 
                image=None, 
                mask=None, 
                modality=None
                )
            
            if is_master():
                self.text_text_banks['text1_features'].append(text1_feature.detach().cpu())
                self.text_text_banks['text2_features'].append(text2_feature.detach().cpu())
                self.text_text_banks['labels'] += label_text
                
            self.fwd_time_m.update(time.time()-end_time)
            end_time = time.time()
            
            prediction = {
                'image_features':text1_feature, 
                'text_features':text2_feature, 
                'logit_scale':self.model.module.temperature.exp()
                }  # 计算梯度需要保留logit_scale为张量
            total_loss = self.loss(prediction, id_list=label_text)   # avg within a batch
        
        self.grad_scaler.scale(total_loss).backward()
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        
        # NOTE: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            self.model.module.temperature.clamp_(0, math.log(self.max_logit_scale))  # exp(t) < 100, t < ln(100)
            
        batch_size = len(pos_text)
        self.loss_m.update(total_loss.item(), batch_size)  
        
        self.bwd_time_m.update(time.time()-end_time)
        end_time = time.time()
    
        with torch.no_grad():
            if is_master():            
                if step % self.eval_step_interval == 0:
                    lr = self.optimizer.param_groups[0]['lr']
                    self.tb_writer.add_scalar('train/loss', self.loss_m.avg, step)
                    self.tb_writer.add_scalar('train/learning_rate', lr, step)
                    self.tb_writer.add_scalar('train/logit_scale', self.model.module.temperature.exp().item(), step)
                    info = f'\nStep {step} | LR {lr:.4e} | Logits Scale {self.model.module.temperature.exp().item():.4f} | Loss {self.loss_m.val:.4f}({self.loss_m.avg:.4f})\n'
                    info += f'Data Time {self.data_time_m.avg:.2f} | FWD Time {self.fwd_time_m.avg:.2f} | BWD Time {self.fwd_time_m.avg:.2f}\n'
                    
                    # eval on dynamic feature bank (trainset)
                    feature_bank1 = torch.concat(self.text_text_banks['text1_features'], dim=0)    # n,d
                    self.text_text_banks['text1_features'] = []
                    feature_bank2 = torch.concat(self.text_text_banks['text2_features'], dim=0)
                    self.text_text_banks['text2_features'] = []
                    id_bank = self.text_text_banks['labels']
                    self.text_text_banks['labels'] = []
                    # cut if the matrix is tooo large
                    if len(id_bank) > 10000:
                        feature_bank1 = feature_bank1[len(id_bank)-10000:, :]
                        feature_bank2 = feature_bank2[len(id_bank)-10000:, :]
                        id_bank = id_bank[len(id_bank)-10000:]
                    logits_num, metrics, top3_pred_idx, gt_rank = get_retrieval_metrics(
                        feature_bank1, feature_bank2, 
                        'text1', 'text2', 
                        self.model.module.temperature.exp().item(), 
                        text_list_or_gt_matrix=id_bank
                        )
                    for name, value in metrics.items():
                        if 'R@' in name:
                            info += f'| {name} {value:.4f} '
                            self.tb_writer.add_scalar(f'train/{name}', value, step)
                    info += '\n'
                        
                    self.loss_m.reset()
                    self.data_time_m.reset()
                    self.fwd_time_m.reset()
                    self.bwd_time_m.reset()
                        
                    # print log
                    print(info)
                    # write log
                    with open(self.log_file, 'a') as f:
                        f.write(info)
                
                # save checkpoint regularly (by epoch)
                if step % self.save_large_interval == 0:
                    torch.save({'step':step,
                                'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),         
                                'text_loss': self.loss_m.avg
                                }, os.path.join(self.checkpoint_dir, f'step_{step}.pth'))
                    print(f'** CHECKPOINT ** save step {step} to step_{step}.pth')
                if step % self.save_small_interval == 0:
                    torch.save({'step':step,
                                'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),         
                                'text_loss': self.loss_m.avg
                                }, os.path.join(self.checkpoint_dir, f'latest_step.pth'))
                    print(f'** CHECKPOINT ** save step {step} to latest_step.pth')