from torch import nn
import torch
import numpy as np
from typing import Tuple
from src.architectures.features_extractors.cnn import Simple_CNN_1D
from src.architectures.features_extractors.transformers import TS_Transformer
from torch import nn
import torch
import numpy as np
from typing import Tuple
from src.architectures.features_extractors.cnn import Simple_CNN_1D

class CNN_Transformer(nn.Module):

    name = "CNN_Transformer"
    
    def __init__(
        self,  
        transformer_config,
        cnn_config, 
        features_size,
        n_classes = 10
    ):
        super(CNN_Transformer, self).__init__()

        self.cnn_config = cnn_config
        self.transformer_config = transformer_config
        self.in_channels = self.cnn_config['in_channels']
        self.features_size_cnn = self.cnn_config['features_size']
        self.features_size_transformer = self.transformer_config['features_size']

        self.conv1 = nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.in_channels, 
                kernel_size=13, 
                padding=6)
        self.bn1 = nn.BatchNorm1d(self.in_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.in_channels, 
                kernel_size=13, 
                padding=6)
        self.bn2 = nn.BatchNorm1d(self.in_channels)
        self.relu2 = nn.ReLU()

        
        self.cnn = Simple_CNN_1D(**cnn_config)
        self.transformer = TS_Transformer(**transformer_config)

        self.features_size = features_size
        self.combiner = nn.Linear(
            self.features_size_cnn + self.features_size_transformer,
            self.features_size
        )
        
        self.classif_head = nn.Linear(self.features_size, n_classes)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))

        x_cnn,_ = self.cnn(x)
        x_transformer,_ = self.transformer(x)
    
        features = self.combiner(torch.cat([x_cnn, x_transformer], dim = 1))

        return features, self.classif_head(features)


        
