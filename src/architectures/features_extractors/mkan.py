import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import torch.nn.functional as F
import numpy as np
import scipy
import copy
import random
#original is here https://github.com/AminMoradiXL/kan_ae/blob/main/ae_kan.py

class Encoder(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        bottleneck_size=20,
        grid_size=16,
        spline_order=4,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.ReLU,
        grid_eps=0.02,
        grid_range=[-1.1, 1.1],
        num_layers = 1,
        n_classes = 16,
        monotonic = False,
        pre_init_file = None,
        polars = False,
        n_layers = 1
    ):
        super(Encoder, self).__init__()
        self.kan_1 = KANLinear(
            input_size,
            hidden_size,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            base_activation=base_activation,
            grid_eps=grid_eps,
            grid_range=grid_range,
            monotonic = monotonic,
            pre_init_file = pre_init_file, 
        )
        additional_layers = [nn.Identity()]
        for i in range(n_layers - 1):
            additional_layers.append(KANLinear(
                hidden_size,
                hidden_size,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
                monotonic = monotonic,
                pre_init_file = pre_init_file, 
            ))
        self.additional_layers = nn.Sequential(*additional_layers)

    def forward(self, x):
        x = x.reshape(x.shape[0],-1)
        x = self.additional_layers(self.kan_1(x))
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        bottleneck_size,
        hidden_size,
        output_size,
        grid_size=5,
        spline_order=4,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
        is_reset_parameters = True,
        step = 1,
        monotonic = False,
        subset =None,
        half = False
    ):
        super(Decoder, self).__init__()
        self.dense = nn.Linear(bottleneck_size, hidden_size)
        self.relu = nn.ReLU()
        self.kan = KANLinear(
            hidden_size,
            output_size,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            base_activation=base_activation,
            grid_eps=grid_eps,
            grid_range=grid_range,
            #is_reset_parameters = True,
            #step = 1,
            monotonic = False,
            #subset =None,
            #half = False
        )

    def forward(self, x):
        x = x
        x = self.kan(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, grid_size=8, monotonic=False, polars = False):
        super(Autoencoder, self).__init__()
        self.encoder = KANLinear(input_size, hidden_size,
                               grid_size = grid_size, monotonic = monotonic, grid_range = [-1,1], polars = polars)
        
        self.decoder = KANLinear(hidden_size, input_size, grid_size = grid_size, monotonic = monotonic, 
                              grid_range = [-3,3])

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.encoder(x)
        features = x
        x = self.decoder(x)
        return features, x


class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=10,
        spline_order=4,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.ReLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
        monotonic_inputs = [],
        monotonic = False,
        polars = False
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_range = grid_range
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.monotonic_inputs = monotonic_inputs
        self.polars = polars
        
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)
        
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        
        
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )
        
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
        self.monotonic = monotonic

        if self.monotonic:
            self.min_init_weight = torch.nn.Parameter(torch.tensor(1.))
            self.mu_w = np.sqrt(
                6*np.pi / (self.in_features * (6*(np.pi-1) + (self.in_features-1)*(3*np.sqrt(3) + 2*np.pi -6)))
            )
            self.sigma_w = np.sqrt(1 / self.in_features)
            self.b_w = - self.mu_w  * self.in_features * np.sqrt(1/2 / np.pi)
            self.base_weight_mono = torch.nn.Parameter( torch.normal(
                mean = np.log(self.mu_w**2) - 1/2*np.log(self.mu_w**2 + self.sigma_w**2),
                std = np.sqrt(np.log(self.mu_w**2 + self.sigma_w**2) - np.log(self.mu_w**2)),
                size = (out_features, in_features)
            ) )
            self.bias_mono = torch.nn.Parameter(
                torch.ones(out_features) * self.b_w
            )
            self.spline_scaler_mono = torch.nn.Parameter(
                torch.zeros(out_features, in_features)
            )
            self.reset_parameters_mono()
        else:
            self.reset_parameters()
        
    def reset_parameters_mono(self):
        init_gamma = torch.randn(self.out_features, self.in_features, self.grid_size + self.spline_order)
        init_gamma[:,:,1:] = torch.cumsum(torch.exp(init_gamma[:,:,1:]), dim = -1)
        init_gamma[:,:,1:] += init_gamma[:,:,0].unsqueeze(-1)
        init_gamma -= (init_gamma.shape[-1]-1) * np.exp(1/2) / 2
        init_gamma /= (init_gamma.shape[-1]-1) * np.exp(1/2) / 2

        input_test = torch.rand(10000, self.in_features) * (self.grid_range[1] - self.grid_range[0]) + self.grid_range[0]
        spline_outputs = F.linear(
                self.b_splines(input_test).view(input_test.size(0), -1),
                init_gamma.view(self.out_features, -1),
            )
        
        self.register_buffer("means", spline_outputs.mean(0))
        self.register_buffer("stds", spline_outputs.std(0))

        base_weight = torch.exp(self.base_weight_mono)
        base_output = F.linear(self.base_activation(input_test), base_weight) + self.bias_mono

        self.register_buffer("means_base", base_output.detach().mean(0))

        spline_weight = torch.zeros(self.out_features, self.in_features, self.grid_size + self.spline_order)
        spline_weight[:,:,0] = init_gamma[:,:,0]
        spline_weight[:,:,1:] = torch.log(init_gamma[:,:,1:] - init_gamma[:,:,:-1])
        
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)
        
        with torch.no_grad():
            self.spline_weight.data.copy_(
                        spline_weight
                    )

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        #assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()
        
    def quad_objective(self, X, A, B, in_features, out_features):
        """
        Computes the quadtratic objective
        
        Args:
            X (np.array): current solution (in_features * out_features * (grid_size + spline_order)).
            A (np.array): output array of shape (batch_size, in_features, grid_size + spline_order).
            B (np.array): b-splines values (batch_size, in_features, out_features)
        
        Returns:
            float: MSE loss of the current solution
        """
        
        X = X.reshape(in_features, out_features, -1)
        Approximation = np.einsum('ijp,jkp->ijk', A, X) # batch_size, in_features, out_features
        loss = np.mean((Approximation - B)**2)
        return loss

    def curve2coeff_mono(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points with monotonic restriction.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """

        #assert x.dim() == 2 and x.size(1) == self.in_features
        
        A = np.array(self.b_splines(x)) # b_s, in_f, grid_size + spline_order
        B = np.array(y)  # b_s, in_f, out_f
        
        in_features = B.shape[1]
        out_features = B.shape[2]
        
        # init of the iterative method
        X_init = np.abs(np.random.rand(in_features, out_features, (self.grid_size + self.spline_order)))
        X_init = (np.cumsum(X_init, -1) - 1/2 * (self.grid_size + self.spline_order)).reshape(-1)
        
        # restriction matrix
        rest_matrix = (
            np.eye(len(X_init)) 
            - 
            np.diag(np.ones(len(X_init)-1), k=1))[[i for i in range(len(X_init)) if (i+1) % (self.grid_size + self.spline_order)]]
        
        # linear constraint
        constraint = scipy.optimize.LinearConstraint(
            rest_matrix,
            -np.inf,
            0,
        )
        
        # iterative method
        optim = scipy.optimize.minimize(
            fun = lambda x: self.quad_objective(x, A, B, in_features, out_features),
            x0 = X_init,
            constraints = constraint,
            method = 'trust-constr',
        )
        
        X = optim.x
        X = X.reshape(in_features, out_features, -1)
        
        return torch.tensor(X).transpose(0,1)

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def scale_spline_weight_mono(self, gamma):
        return gamma * torch.exp(self.spline_scaler_mono.unsqueeze(-1))
    
    def scale_spline_weight(self, gamma):
        return gamma * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        
        x = torch.clamp(x, self.grid_range[0], self.grid_range[1])
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)
        
        if self.monotonic:
            weights = self.spline_weight

            gamma = torch.zeros(weights.shape, device = weights.device)
            gamma[:,:,0] = weights[:,:,0]
            gamma[:,:,1:] = torch.cumsum(torch.exp(weights[:,:,1:]),dim = -1) + gamma[:,:,0].unsqueeze(-1)
            gamma = self.scale_spline_weight_mono(gamma)

            if self.polars:
                #x: r_1, ... r_256, phi_1, ... phi_256
                ids = np.arange(self.in_features // 2)
                
                mask_rad = torch.ones(gamma.shape, device = weights.device)
                mask_rad[0, ids+self.in_features // 2] = 0
                
                mask_phase_1 = torch.ones(gamma.shape, device = weights.device)
                mask_phase_1[1, ids] = 0

                mask_phase_2 = torch.ones(gamma.shape, device = weights.device)
                mask_phase_2[2, ids] = 0
                
                gamma = gamma * mask_rad * mask_phase_1 * mask_phase_2

            spline_output = (F.linear(
                self.b_splines(x).view(x.size(0), -1),
                gamma.view(self.out_features, -1)) - self.means.unsqueeze(0)) / self.stds.unsqueeze(0)
            
            spline_output = spline_output
            base_weight = torch.exp(self.base_weight_mono)

            if self.polars:
                #x: r_1, ... r_256, phi_1, ... phi_256
                ids = np.arange(self.in_features // 2)
                
                mask_rad = torch.ones(base_weight.shape, device = weights.device)
                mask_rad[0, ids+self.in_features // 2] = 0
                
                mask_phase_1 = torch.ones(base_weight.shape, device = weights.device)
                mask_phase_1[1, ids] = 0

                mask_phase_2 = torch.ones(base_weight.shape, device = weights.device)
                mask_phase_2[2, ids] = 0
                
                base_weight = base_weight * mask_rad * mask_phase_1 * mask_phase_2
            
            base_output = F.linear(self.base_activation(x), base_weight) + self.bias_mono #- self.means_base.unsqueeze(0)
            
            output = (spline_output) * self.scale_noise #+ base_output
            #output = base_output
            output = output.view(*original_shape[:-1], self.out_features)
            
            return output

        base_output = F.linear(self.base_activation(x), self.base_weight)

        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output
        
        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

    def get_coefs(self):
        
        weights = self.spline_weight
        gamma = torch.zeros(weights.shape, device = weights.device)
        gamma[:,:,0] = weights[:,:,0]
        gamma[:,:,1:] = torch.cumsum(torch.exp(weights[:,:,1:]),dim = -1) + gamma[:,:,0].unsqueeze(-1)

        return gamma
    
    def get_weights(self, coefs):
        spline_weight = torch.zeros(self.out_features, self.in_features, self.grid_size + self.spline_order, device = coefs.device)
        spline_weight[:,:,0] = coefs[:,:,0]
        spline_weight[:,:,1:] = torch.log(coefs[:,:,1:] - coefs[:,:,:-1])
        
        return spline_weight
        

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)

        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )
