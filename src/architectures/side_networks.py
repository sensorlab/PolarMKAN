from src.architectures.viewmakers import Viewmaker, Viewmaker_1D
import torch.nn as nn
import torch
import numpy as np
from torchvision.transforms import RandomRotation

class Amplifier(nn.Module):
    def __init__(self, in_channels, signal_length, noise_std=0):
        super(Amplifier, self).__init__()
        self.in_channels = in_channels
        self.signal_length = signal_length
        self.noise_std = noise_std

    def forward(self, x):
        scale = torch.randn(x.shape, device = x.device) * self.noise_std + 1
        x *= scale
        return x

class Bias(nn.Module):
    def __init__(self, in_channels, signal_length, noise_std=0):
        super(Bias, self).__init__()
        self.in_channels = in_channels
        self.signal_length = signal_length
        self.noise_std = noise_std

    def forward(self, x):
        bias = torch.randn(1, device = x.device) * self.noise_std
        x += bias
        return x

class Noise(nn.Module):
    def __init__(self, in_channels, signal_length, noise_std=0):
        super(Noise, self).__init__()
        self.in_channels = in_channels
        self.signal_length = signal_length
        self.noise_std = noise_std

    def forward(self, x):
        noise = torch.randn(x.shape, device = x.device) * self.noise_std
        x += noise
        return x

class Inverse(nn.Module):
    def __init__(self, in_channels, signal_length, noise_std=0):
        super(Inverse, self).__init__()
        self.in_channels = in_channels
        self.signal_length = signal_length

    def forward(self, x):
        inverse_indices = [x.shape[-1] - 1 - i for i in range(x.shape[-1])]
        return x[..., inverse_indices]

class Rotation(nn.Module):

    def __init__(self, in_channels, signal_length, noise_std=0):
        super(Rotation, self).__init__()
        self.in_channels = in_channels
        self.signal_length = signal_length
        self.noise_std = noise_std

    def forward(self, x):
        gamma = torch.randn(1, device = x.device) * self.noise_std
        new_x = torch.zeros(x.shape, device = x.device)
        new_x[..., 0, :] = x[..., 0, :] * torch.cos(gamma) - x[..., 1, :] * torch.sin(gamma)
        new_x[..., 1, :] = x[..., 0, :] * torch.sin(gamma) + x[..., 1, :] * torch.cos(gamma)
        return new_x

class Mlp(nn.Module):
    def __init__(self, in_features, out_features, apply_softmax = False):
        super(Mlp, self).__init__()
        self.ln1 = nn.Linear(in_features = in_features, out_features = in_features)
        self.relu = nn.ReLU()
        self.ln2 = nn.Linear(in_features = in_features, out_features = out_features)
        self.apply_softmax = apply_softmax
        self.softmax = nn.Softmax()
        
    def forward(self, x):

        out = self.ln2(self.relu(self.ln1(x)))
        if self.apply_softmax:
            out = self.softmax(out)
        return out
        
class Augmentation_Masked(nn.Module):
    def __init__(self, in_channels, singal_length, aug_module, noise_std = 0.01, num_bits = 4, prob = 0.5):
        super(Augmentation_Masked, self).__init__()

        assert singal_length % num_bits == 0
        self.size = in_channels * singal_length
        self.noise_std = noise_std
        self.num_bits = num_bits
        self.in_channels = in_channels
        self.singal_length = singal_length
        self.bit_size = singal_length // num_bits
        self.prob = prob
        self.layers = nn.ModuleList(
            [   
                aug_module(in_channels, singal_length // num_bits, noise_std)

                for i in range(num_bits)
            ]
        )

    def forward(self, x):
        new_x = torch.zeros(x.shape, device = x.device)
        for i in range(self.num_bits):
            if np.random.choice([0,1], p = [1-self.prob, self.prob]):
                new_x[...,i * self.bit_size: (i+1) * self.bit_size] =\
                self.layers[i](
                    x[...,i * self.bit_size: (i+1) * self.bit_size]
                )
                
            else:
                new_x[...,i * self.bit_size: (i+1) * self.bit_size] =\
                    x[...,i * self.bit_size: (i+1) * self.bit_size]
        
        return new_x


def get_augmentations(viewmaker_config: dict, type:str ='learnable', dims:int = 1):
    
    if type == 'learnable':
         if dims == 1:
            return nn.ModuleList([Viewmaker_1D(**viewmaker_config)])
         if dims == 2:
            return nn.ModuleList([Viewmaker(**viewmaker_config)])
            
    if type == 'static':
        if dims == 1:
            return nn.ModuleList(
                [
                    Augmentation_Masked(**viewmaker_config, aug_module = Noise),
                    Augmentation_Masked(**viewmaker_config, aug_module = Bias),
                    Augmentation_Masked(**viewmaker_config, aug_module = Amplifier),
                    Augmentation_Masked(**viewmaker_config, aug_module = Rotation),
                ]
            )
        else:
            return nn.ModuleList(
                [
                    RandomRotation(0),
                    RandomRotation(degrees = [90,90]),
                    RandomRotation(degrees = [-90,-90]),
                    RandomRotation(degrees = [180,180])
                ]
            )
            
        


    