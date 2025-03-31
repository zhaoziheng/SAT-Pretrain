import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.initialization.weight_init import InitWeights_He

# from .SwinUNETR import SwinUNETR_Enc
# from .umamba_mid import ResidualMambaMidEncoder

class Atlas_Tower(nn.Module):
    """
    Embed 2D slices via ResNet50, and combine pooled features from different slices via Transformer Encoder
    """
    def __init__(self, *, out_dim=768, vision_backbone):
        """
        Args:
            out_dim (int, optional): feature token embedding length. Defaults to 768.
        """
        super().__init__()
        
        self.encoder = {
            # 'SwinUNETR' : SwinUNETR_Enc(
            #                 img_size=[288, 288, 96],    # 48, 48, 96, 192, 384, 768
            #                 in_channels=3,
            #                 feature_size=48,  
            #                 drop_rate=0.0,
            #                 attn_drop_rate=0.0,
            #                 dropout_path_rate=0.0,
            #                 use_checkpoint=False,
            #                 ),
            'UNET' : PlainConvEncoder(
                                   input_channels=3, 
                                   n_stages=6, 
                                   features_per_stage=(64, 64, 128, 256, 512, 768), 
                                   conv_op=nn.Conv3d, 
                                   kernel_sizes=3, 
                                   strides=(1, 2, 2, 2, 2, 2), 
                                   n_conv_per_stage=(2, 2, 2, 2, 2, 2), 
                                   conv_bias=True, 
                                   norm_op=nn.InstanceNorm3d,
                                   norm_op_kwargs={'eps': 1e-5, 'affine': True}, 
                                   dropout_op=None,
                                   dropout_op_kwargs=None,
                                   nonlin=nn.LeakyReLU, 
                                   nonlin_kwargs=None,
                                   return_skips=True,
                                   nonlin_first=False
                                   ),
            'UNET-L' : PlainConvEncoder(
                                   input_channels=3, 
                                   n_stages=6, 
                                   features_per_stage=(128, 128, 256, 512, 1024, 1536), 
                                   conv_op=nn.Conv3d, 
                                   kernel_sizes=3, 
                                   strides=(1, 2, 2, 2, 2, 2), 
                                   n_conv_per_stage=(2, 2, 2, 2, 2, 2), 
                                   conv_bias=True, 
                                   norm_op=nn.InstanceNorm3d,
                                   norm_op_kwargs={'eps': 1e-5, 'affine': True}, 
                                   dropout_op=None,
                                   dropout_op_kwargs=None,
                                   nonlin=nn.LeakyReLU, 
                                   nonlin_kwargs=None,
                                   return_skips=True,
                                   nonlin_first=False
                                   ),
            'UNET-H' : PlainConvEncoder(
                                   input_channels=3, 
                                   n_stages=6, 
                                   features_per_stage=(128, 128, 256, 512, 1024, 1536), 
                                   conv_op=nn.Conv3d, 
                                   kernel_sizes=3, 
                                   strides=(1, 2, 2, 2, 2, 2), 
                                   n_conv_per_stage=(3, 3, 3, 3, 3, 3), 
                                   conv_bias=True, 
                                   norm_op=nn.InstanceNorm3d,
                                   norm_op_kwargs={'eps': 1e-5, 'affine': True}, 
                                   dropout_op=None,
                                   dropout_op_kwargs=None,
                                   nonlin=nn.LeakyReLU, 
                                   nonlin_kwargs=None,
                                   return_skips=True,
                                   nonlin_first=False
                                   ),
            # 'UMamba' : ResidualMambaMidEncoder(
            #             input_channels=3,
            #             n_stages=6,
            #             features_per_stage=(64, 64, 128, 256, 512, 768),
            #             conv_op=nn.Conv3d,
            #             kernel_sizes=3,
            #             strides=(1, 2, 2, 2, 2, 2),
            #             n_blocks_per_stage=(1, 1, 1, 1, 1, 1),
            #             conv_bias=True,
            #             norm_op=nn.InstanceNorm3d,
            #             norm_op_kwargs={'eps': 1e-5, 'affine': True}, 
            #             dropout_op=None,
            #             dropout_op_kwargs=None,
            #             nonlin=nn.LeakyReLU, 
            #             nonlin_kwargs=None,
            #             return_skips=True
            #     )
        }[vision_backbone]
        
        self.encoder.apply(InitWeights_He(1e-2))

        self.projection_layer = {
            # 'SwinUNETR' : nn.Sequential(
            #             nn.Linear(1536, 768),
            #             nn.GELU(),
            #             nn.Linear(768, out_dim),
            #             nn.GELU()
            #         ),
            'UNET' : nn.Sequential(
                        nn.Linear(1792, 768),
                        nn.GELU(),
                        nn.Linear(768, out_dim),
                        nn.GELU()
                    ),
            'UNET-L' : nn.Sequential(
                        nn.Linear(3584, 1536),
                        nn.GELU(),
                        nn.Linear(1536, out_dim),
                        nn.GELU()
                    ),
            'UNET-H' : nn.Sequential(
                        nn.Linear(3584, 1536),  # 128, 128, 256, 512, 1024, 1536 --> 3584 --> 768
                        nn.GELU(),
                        nn.Linear(1536, out_dim),
                        nn.GELU()
                    ),
            # 'UMamba' : nn.Sequential(
            #             nn.Linear(1792, 768),  # 64, 64, 128, 256, 512, 768 --> 768
            #             nn.GELU(),
            #             nn.Linear(768, out_dim),
            #             nn.GELU()
            #         ),
        }[vision_backbone]
        
        self.max_pooling = nn.MaxPool3d(2, 2)
        
    def lock(self):
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, img, mask):
        """
        Pooling attention over area features from each slice
        
        Args:
            img (tensor): (b, c, h, w, depth)
            mask (tensor): (b, n, h, w, depth)
            
        Return:
            roi_feature (tensor): (b, n, d)
        """
 
        multiscale_feature_ls = self.encoder(img)  
        roi_feature_ls = []
        for i, feature_map in enumerate(multiscale_feature_ls): # b d h w d
            feature_map = repeat(feature_map, 'b d h w depth -> b n d h w depth', n=mask.shape[1])
            
            if i > 0:
                mask = self.max_pooling(mask)   # b n h w d
            feature_mask = repeat(mask, 'b n h w depth -> b n d h w depth', d=feature_map.shape[2])
            
            roi_feature = feature_mask * feature_map
            roi_feature = torch.sum(roi_feature, (3, 4, 5)) # b n d
            roi_size = torch.sum(mask, (2, 3, 4)) + 1e-14  # b n
            roi_size = repeat(roi_size, 'b n -> b n one', one=1) # b n 1
            
            roi_feature /= roi_size # b n d
            roi_feature_ls.append(roi_feature)
            
        # concat multi-scale features and project to output dim
        roi_feature = torch.concat(roi_feature_ls, dim=-1)  # b n D
        roi_feature = self.projection_layer(roi_feature)    # b n d
        return roi_feature

if __name__ == '__main__':
    v3d = Atlas_Tower().cuda()
    
    img3d = torch.randn(2, 3, 512, 512, 4).cuda()
    msk3d = torch.ones(2, 512, 512, 4).cuda()
    preds = v3d(img3d, msk3d)
    print("ViT3D output size:", preds.shape)
    
    img2d = torch.randn(2, 3, 512, 512).cuda()
    msk2d = torch.ones(2, 512, 512).cuda()
    preds = v3d(img2d, msk2d)
    print("ViT2D output size:", preds.shape)