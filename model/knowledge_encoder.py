import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

class Knowledge_Encoder(nn.Module):
    """
    Embed 2D and 3D patches via different MLPs, and share a Transformer Encoder
    """
    def __init__(self,
                 atlas_tower,
                 text_tower,
                #  use_pretrained_text_tower=False,
                #  text_tower_checkpoint='',
                #  use_pretrained_unet_encoder=False,
                #  unet_encoder_checkpoint='',
                ):
        super().__init__()
        # LP
        self.text_tower = text_tower
        
        # VLP
        self.atlas_tower = atlas_tower
        self.projection_layer = nn.Sequential(
            nn.Linear(768, 768),
            nn.GELU(),
            nn.Linear(768, 768)
        )
        self.modality_embed = nn.Embedding(5, 768)
        
        self.temperature = nn.Parameter(torch.log(torch.tensor(1/0.07)))
        
        self.init_parameters()
        
        # if use_pretrained_text_tower:
        #     self._use_pretrained_text_tower(text_tower_checkpoint)
        
        # if use_pretrained_unet_encoder:
        #     self._use_pretrained_unet_encoder(unet_encoder_checkpoint)

    # def _use_pretrained_text_tower(self, checkpoint_path):  # initialized with xiaoman bert or text tower (wrapped in another knowledge tower)
    #     print(f'** MODEL ** initial with pretrained text tower from {checkpoint_path}')
    #     cp = torch.load(checkpoint_path)
    #     # judge to load xiaomanbert or text tower ckpt in a knowledge encoder
    #     if 'model_state_dict' in cp:
    #         model_dict = self.state_dict()
    #         state_dict = {k.replace('module.', ''):v for k,v in cp['model_state_dict'].items() if 'atlas_tower' not in k and k.replace('module.', '') in model_dict.keys()}
    #         model_dict.update(state_dict)
    #         self.load_state_dict(model_dict)
    #     else:   # xiaomanbert on umls corpus
    #         if 'bert_model.embeddings.position_ids' in cp['state_dict']:
    #             del cp['state_dict']['bert_model.embeddings.position_ids']
    #         self.text_tower.load_state_dict(cp['state_dict'])
    #         self.temperature = self.text_tower.temperature
        
    # def _use_pretrained_unet_encoder(self, checkpoint_path):
    #     print(f'** MODEL ** initial with pretrained unet encoder from {checkpoint_path}')
    #     cp = torch.load(checkpoint_path)
    #     model_dict =  self.atlas_tower.state_dict()
    #     # find the parameters in unet encoder
    #     state_dict = {k.replace('module.backbone.encoder', 'encoder'):v for k,v in cp['model_state_dict'].items() if 'backbone.encoder' in k and k.replace('module.backbone.encoder', 'encoder') in model_dict.keys()}
    #     model_dict.update(state_dict)
    #     self.atlas_tower.load_state_dict(model_dict)
    
    def init_parameters(self):
        nn.init.constant_(self.temperature, np.log(1 / 0.07))
        for m in self.projection_layer:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=768 ** -0.5)
        
    def lock_text_tower(self):
        self.text_tower.lock()
        
    def lock_atlas_tower(self):
        self.atlas_tower.lock()
        for param in self.projection_layer.parameters():
            param.requires_grad = False
        for param in self.modality_embed.parameters():
            param.requires_grad = False 
    
    def forward(self, text1, text2, image, mask, modality):
        """
        Args:
            text: b
            image: bchwd
            mask: bnhwd
            modality: b (aligned with text)
            
        Returns:
            text_feature: bd
            atlas_feature: bnd
        
        When constrastive between atlas and text, text and modality shoule be (bn)
        
        0. text --> text tower
        1. image + mask --> atlas tower
        2. text1, text2 --> text tower  (need norm)
        3. text1 + modality --> text tower + proj + modality
        4. image + mask + text1 + modality --> atlas tower + text tower + proj + modality   (need norm)
        """
        assert (image == None and mask == None) or (image is not None and mask is not None), \
            "Model Input Error. Image and mask should be given at the same time or both left None" # altas = image + mask
        
        if image is not None:
            atlas_feature = self.atlas_tower(image, mask)   # b n d
            
            if text1 is not None:   # contrast between atlas and (text + modality)
                text_feature = self.text_tower(text1) # bn d
                text_feature = self.projection_layer(text_feature)  # bn d
                modality_feature = self.modality_embed(modality) # bn d
                text_feature = text_feature + modality_feature
                
                text_feature = F.normalize(text_feature, dim=-1)
                atlas_feature = F.normalize(atlas_feature, dim=-1)
                return text_feature, atlas_feature, self.temperature.exp()  # --> 4

            else:   # encode atlas
                return atlas_feature, self.temperature.exp()    # --> 1
            
        elif text2 is not None:   # contrast between text and text
            text_feature1 = self.text_tower(text1)
            text_feature2 = self.text_tower(text2)
            proj_text_feature1 = self.projection_layer(text_feature1)
            proj_text_feature2 = self.projection_layer(text_feature2)           
            
            text_feature1 = F.normalize(text_feature1, dim=-1)
            text_feature2 = F.normalize(text_feature2, dim=-1)
            proj_text_feature1 = F.normalize(proj_text_feature1, dim=-1)
            proj_text_feature2 = F.normalize(proj_text_feature2, dim=-1)
            
            return text_feature1, text_feature2, proj_text_feature1, proj_text_feature2, self.temperature.exp()     # --> 2
        
        elif modality is not None:   # encode text, proj, and plus modality)
            text_feature = self.text_tower(text1)
            text_feature = self.projection_layer(text_feature)
            modality_feature = self.modality_embed(modality)
            text_feature = text_feature + modality_feature
            return text_feature, self.temperature.exp()     # --> 3
            
        else:   # encode text, unnormalized
            text_feature = self.text_tower(text1)    # (n, d)
            proj_text_feature = self.projection_layer(text_feature)
            
            return text_feature, proj_text_feature, self.temperature.exp()     # --> 0
    