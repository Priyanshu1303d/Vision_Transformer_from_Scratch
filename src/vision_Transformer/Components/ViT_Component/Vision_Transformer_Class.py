import torch
import torch.nn as nn
import torch.nn.functional as F
from src.vision_Transformer.Components.ViT_Component.Transformer_EncoderLayer import Transformer_Encoder_Layer
from src.vision_Transformer.Components.ViT_Component.Patch_Embedding import PatchEmbedding


class Vision_Transformer_Class(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, num_classes, embed_dim, num_heads, depth, mlp_dim, dropout_rate):
        super().__init__()

        # Patch embedding: splits image into patches and projects to embedding dimension
        self.patch_embedding = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)

        # Stack multiple transformer encoder layers
        self.encoder_layer = nn.Sequential(*[
            Transformer_Encoder_Layer(embed_dim, num_heads, mlp_dim, dropout_rate)
            for _ in range(depth)
        ])

        # Final normalization before classification
        self.normalization_layer = nn.LayerNorm(embed_dim)

        # Classification head: maps CLS token embedding to class logits
        self.classification_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):  # <-- Added input argument
        # x: (Batch, Channels, Height, Width)
        x = self.patch_embedding(x)  # (Batch, Num_patches+1, Embed_dim)

        x = self.encoder_layer(x)    # (Batch, Num_patches+1, Embed_dim)

        x = self.normalization_layer(x)  # (Batch, Num_patches+1, Embed_dim)

        cls_token = x[:, 0]  # Select CLS token for each batch: (Batch, Embed_dim)
        #means take all the batch , select first row only and select all its column or x[: , 0 , : ]

        x = self.classification_head(cls_token)  # Get class logits

        return x  # (Batch, Num_classes)
