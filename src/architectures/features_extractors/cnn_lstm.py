from torch import nn
import torch
import numpy as np
from typing import Tuple
from src.architectures.features_extractors.cnn import Simple_CNN_1D


class CNN_LSTM(nn.Module):

    """
    Inspired by 
    https://ieeexplore.ieee.org/abstract/document/10615420
    """

    name = "CNN_LSTM"

    def __init__(
        self,  
        lstm_config,
        cnn_config, 
        features_size,
        n_classes = 10
    ):
        super(CNN_LSTM, self).__init__()

        self.cnn_config = cnn_config
        self.lstm_config = lstm_config

        self.in_channels = self.cnn_config['in_channels']
        
        self.features_size_cnn = self.cnn_config['features_size']
        self.features_size_lstm = self.lstm_config['hidden_size']
        self.lstm_input_size = self.lstm_config['input_size']

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
        self.lstm = nn.LSTM(**lstm_config)

        self.features_size = features_size

        self.combiner = nn.Linear(
            self.features_size_cnn + self.features_size_lstm,
            self.features_size
        )
        self.classif_head = nn.Linear(self.features_size, n_classes)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        
        x_cnn,_ = self.cnn(x)
        x = x.permute(0,1,2)
        x = x.reshape(
            x.shape[0], 
            -1, self.lstm_input_size
        )

        x_lstm,_ = self.lstm(x)
        x_lstm = x_lstm.mean(1)

        features = self.combiner(torch.cat([x_cnn, x_lstm], dim = 1))

        return features, self.classif_head(features)

        
