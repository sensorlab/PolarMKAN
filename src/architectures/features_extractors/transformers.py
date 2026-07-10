from torch import nn
import torch
import numpy as np
from typing import Tuple


class TS_Transformer(nn.Module):

    name = "TS_Transformer"

    def __init__(
        self,  
        input_signal_length:int=1024, 
        in_channels:int=2,
        token_size = 10,
        num_layers = 3,
        num_heads = 8,
        features_size:int=100,
        n_classes = 10
    ):
        super(TS_Transformer, self).__init__()

        self.input_signal_length = input_signal_length
        self.in_channels = in_channels
        self.token_size = token_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.features_size = features_size
        self.hidden_size = self.in_channels * self.token_size
        self.num_tokens = self.input_signal_length // self.token_size
        self.n_classes = n_classes
        self.first_embeding = nn.Linear(self.hidden_size, self.hidden_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=self.num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)
        self.final_mlp = nn.Linear(self.hidden_size, self.features_size)
        self.classification_head = nn.Linear(self.features_size, self.n_classes)

    def forward(self, x):
        x = x.swapaxes(1,2)
        x = x.reshape(x.shape[0], self.num_tokens, -1)
        x = x.swapaxes(0,1)
        
        x = self.first_embeding(x)
        
        cls_token = torch.zeros(x[0].shape, device=x.device)
        x = torch.cat([x, cls_token[None, :, :]], dim = 0)

        x = self.transformer_encoder(x)
        x = self.final_mlp(x[0])

        return x, self.classification_head(x)

        

        
        
        

        
        
