from torch import nn
import torch
import numpy as np
from typing import Tuple

class Simple_CNN_1D(nn.Module):
    """
    Simple 1D CNN without resudial connections. 

    Like in L. Milosheski, M. Mohorčič and C. Fortuna, "Spectrum Sensing With Deep Clustering: Label-Free Radio Access Technology Recognition," 
    in IEEE Open Journal of the Communications Society, vol. 5, pp. 4746-4763, 2024, doi: 10.1109/OJCOMS.2024.3436601
    """
    
    def __init__(
        self, 
        layers_output_sizes:list=[16, 32, 64, 128], 
        input_signal_length:int=256, 
        in_channels:int=2,  
        kernel_size:int=3,
        padding_size:int=0,
        features_size:int=100,
        maxplool_strides:int = 2,
        n_classes = 10,
        svd_init = False
    ):
        
        super(Simple_CNN_1D, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.padding_size = padding_size
        self.input_signal_length = input_signal_length
        self.layers_output_sizes = layers_output_sizes
        self.features_size = features_size
        
        self.feature_layers = self._build_feature_layers(
            layers_output_sizes=layers_output_sizes, 
            input_signal_length=input_signal_length, 
            in_channels=in_channels,  
            kernel_size=kernel_size, 
            padding_size=padding_size,
            features_size=features_size,
            maxplool_strides = maxplool_strides
        )

        self.classif_head = nn.Sequential(
            nn.Linear(features_size, features_size), nn.ReLU(), nn.Linear(features_size, n_classes)
        )
        self.svd_init = svd_init
        self.svd_init_layer = nn.Linear(input_signal_length * in_channels, features_size)
        self.svd_init_layer.weight.requires_grad = False
        self.svd_init_layer.bias.requires_grad = False
        
    def _compute_conv1d_output_length(
        self,
        input_length:int,
        padding_size:int=0,
        kernel_size:int=3,
        stride:int=1
    ):
        """
        Compute the output's signal length of the shape (output_channels, output_length)
        """
        output_length = (
            (input_length + 2 * padding_size - kernel_size) // stride  + 1
        )
        return int(output_length)
    
    def _compute_maxpool1d_output_length(
        self,
        input_length:int,
        padding_size:int=0,
        kernel_size:int=2,
        stride:int=2
    ):
        """
        Compute the output's signal length of the shape (output_channels, output_length)
        """
        output_length = (
            (input_length + 2 * padding_size - kernel_size) // stride  + 1
        )
        return int(output_length)
    
        
    def _build_feature_layers(
        self,
        layers_output_sizes:list=[16, 32, 64, 128], 
        input_signal_length:int=1024, 
        in_channels:int=1,  
        kernel_size:int=3, 
        padding_size:int=0,
        features_size:int=100,
        maxplool_strides=4)->nn.Sequential:
        """
        Build the feature map
        """
        # calc the result shape
        cur_signal_length = input_signal_length
        num_layers = len(layers_output_sizes)
        layers = []
        
        for i in range(num_layers):
            # input channels for current layer
            if i == 0:
                cur_in_channels = in_channels
            else:
                cur_in_channels = layers_output_sizes[i-1]
                
            # conv1d->batchnorm->relu->maxpool
            layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=cur_in_channels,
                        out_channels= layers_output_sizes[i], 
                        kernel_size=kernel_size, 
                        padding=padding_size),
                    nn.ReLU(),
                    nn.BatchNorm1d( layers_output_sizes[i]),
                    nn.MaxPool1d(kernel_size=maxplool_strides, stride = maxplool_strides),
                    nn.Conv1d(
                        in_channels=layers_output_sizes[i],
                        out_channels= layers_output_sizes[i], 
                        kernel_size=kernel_size, 
                        padding=padding_size),
                    nn.ReLU(),
                    nn.BatchNorm1d( layers_output_sizes[i]),
                )
            )
            
            # update the singal length
            cur_signal_length = self._compute_conv1d_output_length(
                input_length=cur_signal_length, kernel_size = kernel_size
            )
            cur_signal_length = self._compute_maxpool1d_output_length(
                input_length=cur_signal_length, stride=maxplool_strides, kernel_size=maxplool_strides
            )
            cur_signal_length = self._compute_conv1d_output_length(
                input_length=cur_signal_length, kernel_size = kernel_size
            )
            
        # total size of the output tensor(num_chanhnels, signal_length)
        output_size = cur_signal_length * layers_output_sizes[-1] # default 512
        
        # linear layer for feature map reduction
        layers.extend(
            [
            nn.Flatten(),
            nn.Linear(output_size, features_size),
            ]
        )
        return nn.ModuleList(layers)
        
    def forward(self, x):
        features = x
        for i in range(len(self.feature_layers)):
            features = self.feature_layers[i](features)
        size = self.input_signal_length * self.in_channels
        features = features + self.svd_init_layer(x.reshape(-1,size)) * self.svd_init
        return features, features



class Simple_CNN_2D(nn.Module):
    """
    Simple 2D CNN. 
    
    Adapted from Simple_CNN_1D for image inputs.
    """
    
    def __init__(
        self, 
        layers_output_sizes:list=[16, 32, 64, 128], 
        input_image_size:tuple=(64, 64),  # (H, W)
        in_channels:int=1,                # RGB images by default
        kernel_size:int=3,
        padding_size:int=0,
        features_size:int=20,
        maxpool_strides:int=2,
        n_classes:int=10,
        svd_init = False
    ):
        super(Simple_CNN_2D, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.padding_size = padding_size
        self.input_image_size = input_image_size
        self.layers_output_sizes = layers_output_sizes
        self.features_size = features_size
        self.svd_init = svd_init
        self.svd_init_layer = nn.Linear(
            input_image_size[0] * input_image_size[1] * in_channels, features_size)
        
        self.svd_init_layer.weight.requires_grad = False
        self.svd_init_layer.bias.requires_grad = False

        # first small conv block (like in your 1D version)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.bn3 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()

        # Feature extraction layers
        self.feature_layers = self._build_feature_layers(
            layers_output_sizes=layers_output_sizes, 
            input_image_size=input_image_size, 
            in_channels=in_channels,  
            kernel_size=kernel_size, 
            padding_size=padding_size,
            features_size=features_size,
            maxpool_strides=maxpool_strides
        )

        # Classifier head
        self.classif_head = nn.Sequential(
            nn.Linear(features_size, features_size),
            nn.ReLU(),
            nn.Linear(features_size, n_classes)
        )
        

    def _compute_conv2d_output_size(self, h:int, w:int, kernel:int, stride:int=1, padding:int=0):
        """Compute output (H, W) after Conv2d"""
        h_out = (h + 2*padding - kernel)//stride + 1
        w_out = (w + 2*padding - kernel)//stride + 1
        return h_out, w_out

    def _compute_maxpool2d_output_size(self, h:int, w:int, kernel:int, stride:int, padding:int=0):
        """Compute output (H, W) after MaxPool2d"""
        h_out = (h + 2*padding - kernel)//stride + 1
        w_out = (w + 2*padding - kernel)//stride + 1
        return h_out, w_out

    def _build_feature_layers(
        self,
        layers_output_sizes:list=[16, 32, 64, 128], 
        input_image_size:tuple=(64, 64), 
        in_channels:int=3,  
        kernel_size:int=7, 
        padding_size:int=3,
        features_size:int=100,
        maxpool_strides:int=2
    ):
        """Build convolutional feature extractor for images"""
        cur_h, cur_w = input_image_size
        self.output_layers_size = [(cur_h, cur_w)]
        
        layers = []
        
        for i, out_channels in enumerate(layers_output_sizes):
            cur_in_channels = in_channels if i == 0 else layers_output_sizes[i-1]
            
            layers.append(
                nn.Sequential(
                    nn.Conv2d(cur_in_channels, out_channels, kernel_size=kernel_size, padding=padding_size),
                    nn.ReLU(),
                    nn.BatchNorm2d(out_channels),
                    nn.MaxPool2d(kernel_size=maxpool_strides, stride=maxpool_strides)
                )
            )
            
            # update spatial dimensions
            cur_h, cur_w = self._compute_conv2d_output_size(cur_h, cur_w, kernel_size, 1, padding_size)
            cur_h, cur_w = self._compute_maxpool2d_output_size(cur_h, cur_w, maxpool_strides, maxpool_strides)
            self.output_layers_size.append((cur_h, cur_w))
        
        # Flatten and linear feature reduction
        output_size = cur_h * cur_w * layers_output_sizes[-1]
        print(f"Final flattened feature size: {output_size}")

        layers.extend([
            nn.Flatten(),
            nn.Linear(output_size, features_size)
        ])
        return nn.ModuleList(layers)

    def forward(self, x):
        features = x
        for layer in self.feature_layers:
            features = layer(features)
        x = x.reshape(x.shape[0],-1)
        
        features = features/10 + self.svd_init_layer(x) * self.svd_init
        return features, features
    
class AE_CNN_1D(nn.Module):

    """
    AE like in Like in L. Milosheski, M. Mohorčič and C. Fortuna, "Spectrum Sensing With Deep Clustering: Label-Free Radio Access Technology Recognition," 
    in IEEE Open Journal of the Communications Society, vol. 5, pp. 4746-4763, 2024, doi: 10.1109/OJCOMS.2024.3436601. 
    """
    
    name = "AE_CNN_1D"
    
    def __init__(
        self, 
        layers_output_sizes:list=[16, 32, 64, 128], 
        input_signal_length:int=1024, 
        in_channels:int=2,  
        kernel_size_conv:int=3, 
        kernel_size_maxpool:int=2,
        n_classes:int=10, 
        padding_size:int=0,
        features_size:int=10):
        
        super(AE_CNN_1D, self).__init__()
        
        self.in_channels = in_channels
        self.kernel_size_conv = kernel_size_conv
        self.kernel_size_maxpool = kernel_size_maxpool
        self.padding_size = padding_size
        self.input_signal_length = input_signal_length
        self.layers_output_sizes = layers_output_sizes
        self.n_classes = n_classes
        self.features_size = features_size

        # build feature map layers and memorize maxpoolings positions
        (self.feature_layers, self.maxpooling_layers_positions, 
        self.internal_signal_length, self.output_size) = self._build_feature_layers(
            layers_output_sizes=layers_output_sizes, 
            input_signal_length=input_signal_length, 
            in_channels=in_channels,
            kernel_size_conv=kernel_size_conv,
            kernel_size_maxpool=kernel_size_maxpool,
            n_classes=n_classes, 
            padding_size=padding_size,
            features_size=features_size
        )
        self.mlp = nn.Sequential(nn.Flatten(), nn.ReLU(), nn.Linear(self.output_size, self.features_size))
        # build reconstruction layers: inverse order of feature map layers
        # memorize maxunpoolings positions
        self.reconstruction_layers, self.maxunpooling_layers_positions = \
        self._build_reconstruction_layers(
            layers_output_sizes=layers_output_sizes, 
            output_signal_length=self.internal_signal_length, 
            in_channels=in_channels,  
            kernel_size_conv=kernel_size_conv,
            kernel_size_maxpool=kernel_size_maxpool,
            n_classes=n_classes, 
            padding_size=padding_size,
            features_size=features_size
        )
        
    def _compute_conv1d_output_length(
        self,
        input_length:int,
        padding_size:int=3,
        kernel_size:int=7,
        stride:int=1
    )->int:
        """
        Compute the output's signal length of the shape (output_channels, output_length)
        """
        output_length = (
            (input_length + 2 * padding_size - kernel_size) // stride  + 1
        )
        return int(output_length)
    
    def _compute_maxpool1d_output_length(
        self,
        input_length:int,
        padding_size:int=0,
        kernel_size:int=4,
        stride:int=4
    )->int:
        """
        Compute the output's signal length of the shape (output_channels, output_length)
        """
        output_length = (
            (input_length + 2 * padding_size - kernel_size) // stride  + 1
        )
        return int(output_length)
    
        
    def _build_feature_layers(
        self,
        layers_output_sizes:list=[16, 32, 64, 128], 
        input_signal_length:int=1024, 
        in_channels:int=1,  
        kernel_size_conv:int=7, 
        kernel_size_maxpool:int=4, 
        n_classes:int=10, 
        padding_size:int=3,
        features_size:int=10)->Tuple[nn.ModuleList, set, int]:
        """
        Build the feature map, return maxpooling layers indices and internal signal length
        """
        # calc the result shape
        cur_signal_length = input_signal_length
        
        num_layers = len(layers_output_sizes)
        
        layers = []
        maxpooling_layers_positions = set()
        
        counter = 0
        
        for i in range(num_layers):
            
            # input channels for current layer
            if i == 0:
                cur_in_channels = in_channels
            else:
                cur_in_channels = layers_output_sizes[i-1]

            # memorize current maxpooling's position
            maxpooling_layers_positions.add(counter + 4)
            
            counter += 5

            # conv1d->batchnorm->relu->maxpool
            layers.extend(
                    [
                    nn.Conv1d(
                        in_channels=cur_in_channels,
                        out_channels= layers_output_sizes[i], 
                        kernel_size=kernel_size_conv, 
                        padding=padding_size),
                    nn.Dropout(p=0.2),
                    nn.BatchNorm1d(layers_output_sizes[i]),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=kernel_size_maxpool,stride=kernel_size_maxpool,
                                 return_indices=True)
                    ]
            )

            # update the singal length
            cur_signal_length = self._compute_conv1d_output_length(
                input_length=cur_signal_length,kernel_size=kernel_size_conv, padding_size = padding_size
            )
            cur_signal_length = self._compute_maxpool1d_output_length(
                input_length=cur_signal_length,kernel_size=kernel_size_maxpool,stride=kernel_size_maxpool
            )
            
        # total size of the output tensor(num_chanhnels, signal_length)
        output_size = cur_signal_length * layers_output_sizes[-1] # default 512

        
        return (nn.ModuleList(layers), maxpooling_layers_positions, cur_signal_length, output_size)
    
    def _build_reconstruction_layers(
        self,
        layers_output_sizes:list=[16, 32, 64, 128], 
        output_signal_length:int=512, 
        in_channels:int=1,  
        kernel_size_conv:int=7, 
        kernel_size_maxpool:int=4, 
        n_classes:int=10, 
        padding_size:int=3,
        features_size:int=10)->Tuple[nn.ModuleList, set]:
        """
        Reconstruction from embedings
        """
        
        num_layers = len(layers_output_sizes)
        layers_output_sizes = layers_output_sizes[::-1]
        
        layers = []
        maxunpooling_layers_positions = set()

        # inverse order: linear layer first
        
        
        counter = 0
        
        for i in range(num_layers):
            
            # cur out channels
            if i == num_layers - 1:
                cur_out_channels = in_channels
            else:
                cur_out_channels = layers_output_sizes[i+1]

            # memorize cur maxunpool position
            maxunpooling_layers_positions.add(counter + 1)
            counter += 5

            # batchnorm->maxunpool->conv1transpose->relu
            layers.extend(
                    [
                    nn.BatchNorm1d(layers_output_sizes[i]),
                    nn.MaxUnpool1d(kernel_size=kernel_size_maxpool),
                    nn.ConvTranspose1d(
                        in_channels=layers_output_sizes[i],
                        out_channels=cur_out_channels, 
                        kernel_size=kernel_size_conv, 
                        padding=padding_size),
                    nn.Dropout(p=0.2),
                    nn.ReLU()
                    ]
            )
        
        return nn.ModuleList(layers), maxunpooling_layers_positions
        
    def forward(self, x, aug = None, aug_index = None)->Tuple[torch.Tensor, torch.Tensor]:
        """
        Firstly obtain feature map, then reconstruct the initial image from it.
        
        Indices from maxunpoolings correspond for those from maxpoolings.
        
        """
        maxpooling_indices = []
        sizes = []
        for i, layer in enumerate(self.feature_layers):
            sizes.append(x.shape)
            if i in self.maxpooling_layers_positions:
                x, indices = layer(x)
                maxpooling_indices.append(indices)
            else:
                x = layer(x)
                
        features = self.mlp(x)
        sizes = sizes[::-1]
        counter = 0
        maxpooling_indices = maxpooling_indices[::-1]
        
        for i, layer in enumerate(self.reconstruction_layers):
            if i in self.maxunpooling_layers_positions:
                x = layer(x, maxpooling_indices[counter], output_size = sizes[i])
                counter += 1
            else:
                x = layer(x)
                
        return features, x



class AE_CNN_2D(nn.Module):    
    name = "AE_CNN_2D"
    
    def __init__(
        self, 
        layers_output_sizes:list=[16, 32, 64, 128], 
        input_signal_w:int=60,
        input_signal_h:int=60, 
        in_channels:int=1,  
        kernel_size_conv:int=3, 
        kernel_size_maxpool:int=2,
        n_classes:int=10, 
        padding_size:int=0,
        features_size:int=10,
        svd_init=False):
        
        super(AE_CNN_2D, self).__init__()

        self.svd_init = svd_init
        self.svd_init_layer = nn.Linear(input_signal_w * input_signal_h, features_size)
        self.svd_init_layer.weight.requires_grad = False
        self.svd_init_layer.bias.requires_grad = False
        
        self.in_channels = in_channels
        self.kernel_size_conv = kernel_size_conv
        self.kernel_size_maxpool = kernel_size_maxpool
        self.padding_size = padding_size
        self.input_signal_w = input_signal_w
        self.input_signal_h = input_signal_h
        self.layers_output_sizes = layers_output_sizes
        self.n_classes = n_classes
        self.features_size = features_size

        # build feature map layers and memorize maxpoolings positions
        (self.feature_layers, self.maxpooling_layers_positions, 
        self.internal_signal_length, self.output_size) = self._build_feature_layers(
            layers_output_sizes=layers_output_sizes, 
            input_signal_w=input_signal_w, 
            input_signal_h = input_signal_h,
            in_channels=in_channels,
            kernel_size_conv=kernel_size_conv,
            kernel_size_maxpool=kernel_size_maxpool,
            n_classes=n_classes, 
            padding_size=padding_size,
            features_size=features_size
        )
        self.mlp = nn.Sequential(nn.Flatten(), nn.ReLU(), nn.Linear(self.output_size, self.features_size))
        # build reconstruction layers: inverse order of feature map layers
        # memorize maxunpoolings positions
        self.reconstruction_layers, self.maxunpooling_layers_positions = \
        self._build_reconstruction_layers(
            layers_output_sizes=layers_output_sizes, 
            output_signal_length=self.internal_signal_length, 
            in_channels=in_channels,  
            kernel_size_conv=kernel_size_conv,
            kernel_size_maxpool=kernel_size_maxpool,
            n_classes=n_classes, 
            padding_size=padding_size,
            features_size=features_size
        )
        
    def _compute_conv1d_output_length(
        self,
        input_length:int,
        padding_size:int=3,
        kernel_size:int=7,
        stride:int=1
    )->int:
        """
        Compute the output's signal length of the shape (output_channels, output_length)
        """
        output_length = (
            (input_length + 2 * padding_size - kernel_size) // stride  + 1
        )
        return int(output_length)
    
    def _compute_maxpool1d_output_length(
        self,
        input_length:int,
        padding_size:int=0,
        kernel_size:int=4,
        stride:int=4
    )->int:
        """
        Compute the output's signal length of the shape (output_channels, output_length)
        """
        output_length = (
            (input_length + 2 * padding_size - kernel_size) // stride  + 1
        )
        return int(output_length)
    
        
    def _build_feature_layers(
        self,
        layers_output_sizes:list=[16, 32, 64, 128], 
        input_signal_w:int=1024, 
        input_signal_h = 100,
        in_channels:int=1,  
        kernel_size_conv:int=3, 
        kernel_size_maxpool:int=2, 
        n_classes:int=10, 
        padding_size:int=0,
        features_size:int=10)->Tuple[nn.ModuleList, set, int]:
        """
        Build the feature map, return maxpooling layers indices and internal signal length
        """
        # calc the result shape
        cur_signal_w = input_signal_w
        cur_signal_h = input_signal_h
        num_layers = len(layers_output_sizes)
        
        layers = []
        maxpooling_layers_positions = set()
        counter = 0
        
        for i in range(num_layers):
            # input channels for current layer
            if i == 0:
                cur_in_channels = in_channels
            else:
                cur_in_channels = layers_output_sizes[i-1]

            # memorize current maxpooling's position
            maxpooling_layers_positions.add(counter + 3)
            counter += 4
            
            # conv1d->batchnorm->relu->maxpool
            layers.extend(
                    [
                    nn.Conv2d(
                        in_channels=cur_in_channels,
                        out_channels= layers_output_sizes[i], 
                        kernel_size=kernel_size_conv, 
                        padding=padding_size),
                    nn.BatchNorm2d(layers_output_sizes[i]),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=kernel_size_maxpool,stride=kernel_size_maxpool,
                                 return_indices=True),
                    ]
            )

            # update the singal length
            cur_signal_h = self._compute_conv1d_output_length(
                input_length=cur_signal_h,kernel_size=kernel_size_conv, padding_size = padding_size
            )
            cur_signal_w = self._compute_conv1d_output_length(
                input_length=cur_signal_w,kernel_size=kernel_size_conv, padding_size = padding_size
            )
            
            cur_signal_h = self._compute_maxpool1d_output_length(
                input_length=cur_signal_h,kernel_size=kernel_size_maxpool,stride=kernel_size_maxpool
            )
            cur_signal_w = self._compute_maxpool1d_output_length(
                input_length=cur_signal_w,kernel_size=kernel_size_maxpool,stride=kernel_size_maxpool
            )

            
        # total size of the output tensor(num_chanhnels, signal_length)
        output_size = cur_signal_h * cur_signal_w * layers_output_sizes[-1] # default 512
        
        return (nn.ModuleList(layers), maxpooling_layers_positions, -1, output_size)
    
    def _build_reconstruction_layers(
        self,
        layers_output_sizes:list=[16, 32, 64, 128], 
        output_signal_length:int=512, 
        in_channels:int=1,  
        kernel_size_conv:int=7, 
        kernel_size_maxpool:int=4, 
        n_classes:int=10, 
        padding_size:int=3,
        features_size:int=10)->Tuple[nn.ModuleList, set]:
        """
        Reconstruction from embedings
        """
        
        num_layers = len(layers_output_sizes)
        layers_output_sizes = layers_output_sizes[::-1]
        layers = []
        maxunpooling_layers_positions = set()

        # inverse order: linear layer first
        counter = 0
        
        for i in range(num_layers):
            # cur out channels
            if i == num_layers - 1:
                cur_out_channels = in_channels
            else:
                cur_out_channels = layers_output_sizes[i+1]

            # memorize cur maxunpool position
            maxunpooling_layers_positions.add(counter + 1)
            counter += 4

            # batchnorm->maxunpool->conv1transpose->relu
            layers.extend(
                    [
                    nn.BatchNorm2d(layers_output_sizes[i]),
                    nn.MaxUnpool2d(kernel_size=kernel_size_maxpool),
                    nn.ConvTranspose2d(
                        in_channels= layers_output_sizes[i],
                        out_channels= cur_out_channels, 
                        kernel_size=kernel_size_conv, 
                        padding=padding_size),
                    nn.ReLU(),
                    ]
            )
        
        return nn.ModuleList(layers), maxunpooling_layers_positions
        
    def forward(self, x, aug = None, aug_index = None)->Tuple[torch.Tensor, torch.Tensor]:
        """
        Firstly obtain feature map, then reconstruct the initial image from it.
        
        Indices from maxunpoolings correspond for those from maxpoolings.
        
        """
        input_x = x.reshape(x.shape[0],-1)
        maxpooling_indices = []
        sizes = []
        for i, layer in enumerate(self.feature_layers):
            if i in self.maxpooling_layers_positions:
                sizes.append(x.shape)
                x, indices = layer(x)
                maxpooling_indices.append(indices)
            else:
                x = layer(x)

        
        features = self.mlp(x) + self.svd_init_layer(input_x) * self.svd_init
        sizes = sizes[::-1]
        counter = 0
        maxpooling_indices = maxpooling_indices[::-1]
        for i, layer in enumerate(self.reconstruction_layers):
            if i in self.maxunpooling_layers_positions:
                x = layer(x, maxpooling_indices[counter], output_size = sizes[counter])
                counter += 1
            else:
                x = layer(x)
                
        return features, x
