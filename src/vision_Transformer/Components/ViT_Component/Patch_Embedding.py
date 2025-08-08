import torch 
from src.vision_Transformer.logging import logger
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, image_size , patch_size , in_channels , embed_dim):
        super().__init__()

        self.patch_size = patch_size
        self.projection = nn.Conv2d(
            in_channels= in_channels,
            out_channels= embed_dim,
            kernel_size= patch_size,
            stride= patch_size
        )

        num_patches = (image_size // patch_size) ** 2
        self.pos_encoding = nn.Parameter(torch.randn(1 , 1 + num_patches , embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1 , 1 , embed_dim))

    def forward(self, x : torch.Tensor):
        # X is (Batch, Channels, Height, Width).
        B , _ , _ , _ = x.size() # we want only the Batch

        x = self.projection(x) # (Batch, Embed_dim, Patch_rows, Patch_cols)

        x = x.flatten(2).transpose(1 , 2) # (Batch, Num_patches, Embed_dim),

        # before cls token was random number b/w 0 to 1 and its dim was (1 , 1 , 768)
        cls_token = self.cls_token.expand(B , -1 , -1) 
        # here -1 means don't change its dimension only channge Batch on the first position (B , 1 , 768)

        x = torch.concat((x , cls_token) , dim= 1) 
        # cls_token (B , 1 , 768)
        # x         (B , 196 , 768)
        # final x = (B , 197 , 768) i.e only add 1 + 196 as they are dimension 1...

        x = x + self.pos_encoding
        # pos encoding = (1 , 197 , 768)
        # after broadcasting done by  pytorch the final shape of pos encoding is (B , 197 , 768)
        # final x      = (B , 197 , 768) as it does element wise addition but the dimesnion remains the same

        return x     # shape : (B , 197 , 768)



