import torch 
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self , in_features , hidden_features, dropout_rate):
        super().__init__()

        self.fc1 = nn.Linear(in_features= in_features , out_features= hidden_features)

        self.fc2 = nn.Linear(in_features= hidden_features , out_features= in_features)

        self.dropout = nn.Dropout(p = dropout_rate)

    def forward(self , x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.dropout(x)

        # GELU is applied after fc1 for non-linearity. --->dding a non-linear activation lets the network capture more complex, real-world data.
        
        # fc2 is linear to preserve the residual structure. --->  The output is just a weighted sum (plus bias), so it remains a linear transformation.
        return x