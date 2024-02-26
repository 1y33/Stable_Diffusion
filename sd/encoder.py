import torch
import torch.nn as nn
from torch.nn import functional as F

from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # (batch_size,CHaneel,Height,Width) - > (batch_size, 128 , height ,width)
            nn.Conv2d(3,128,kernel_size=3,padding=1),

            VAE_ResidualBlock(128,128),
            VAE_ResidualBlock(128,128),

            # H/2 W/2
            nn.Conv2d(128,128,3,2,0),

            VAE_ResidualBlock(128, 256),
            VAE_ResidualBlock(256, 256),

            # H/4 @/4
            nn.Conv2d(256,256,3,2,0),

            VAE_ResidualBlock(256, 512),
            VAE_ResidualBlock(512, 512),

            # h/8 w/8
            nn.Conv2d(512,512,3,2,0),

            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),


            VAE_AttentionBlock(512),

            VAE_ResidualBlock(512, 512),

            nn.GroupNorm(32,512),
            nn.SiLU(),

            # 8 H/8 W/8
            nn.Conv2d(512,8,kernel_size=3,padding=1),

            nn.Conv2d(8,8,kernel_size=1,padding=0),
        )

    def forward(self, x:torch.Tensor,noise:torch.Tensor) -> torch.Tensor :
        # X ( BS ,channels ,height ,width ,(512)
        # noise (BS , Out_channels,H/8, W/8 )

        for module in self:
            if getattr(module, 'stride', None) == (2,2):
                X =F.pad(x,(0,1,0,1))
            x = module(x)

        mean, log_variance =  torch.chunk(x,chunks=2,dim=1)
        log_variance = torch.clamp(log_variance,-30,20)
        variance = torch.exp(log_variance)

        stdev = variance.sqrt()

        x = mean+ stdev*noise

        # Scale output constant
        x *= 0.18215

        return x 