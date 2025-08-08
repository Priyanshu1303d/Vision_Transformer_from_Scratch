import torch
import torch.nn as nn
import torch.nn.functional as F
from src.vision_Transformer.Components.ViT_Component.MLP import MLP



class Transformer_Encoder_Layer(nn.Module):
    def __init__(self , embed_dim , num_heads , mlp_dim , dropout_rate):
        super().__init__()

        self.normalization_layer_1 = nn.LayerNorm(embed_dim)

        self.multi_head_attention = nn.MultiheadAttention(embed_dim=embed_dim , num_heads= num_heads , dropout= dropout_rate , batch_first= True)
        #batch first tells the layer our input data will be in the (Batch, Sequence, Embedding) format.

        self.mlp = MLP(embed_dim , mlp_dim , dropout_rate)

        self.normalization_layer_2 = nn.LayerNorm(embed_dim)

    def forward(self , x):
        residual_1 = x

        x = self.normalization_layer_1(x)

        x , _ = self.multi_head_attention(x)
        # The MultiheadAttention layer returns two things: the processed data and the attention weights. We only need the data, so we take the first item (x, _ or [0]).

        x = x + residual_1

        residual_2 = x
        
        x = self.normalization_layer_2(x)

        x = self.mlp(x)

        x = x + residual_2

        return x



