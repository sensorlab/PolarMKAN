import torch
from torch import nn
from torchvision.models import vit_h_14
from torchvision.models import vit_b_16

class Vit_14(nn.Module):
    def __init__(
        self,
        image_size, 
        num_classes=10
    ):
        super().__init__()
        self.model = vit_h_14(image_size = image_size, num_classes = num_classes)
        self.dense = self.model.heads
        self.image_size = image_size
        self.f_map = self.model.encoder
        
    def forward(
        self,
        x
    ):
        x = x[:,[0],:]
        x = x.expand((x.shape[0],3, self.image_size, self.image_size))
        x = self.model._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.model.encoder(x)
        f = x[:, 0]
        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.model.heads(x)

        return f, x