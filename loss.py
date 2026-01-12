import torch
import torch.nn as nn
from torchvision.models import vgg19
import pytorch_lightning as pl


def custom_sigmoid(x, k=1):
    """
    Custom sigmoid function with a steepness parameter k.

    Input: torch.Tensor
        x: Tensor to apply the sigmoid function to.
        k: Steepness of the sigmoid curve. Default is 1.

    Output: torch.Tensor
        Tensor with the same shape as x, where each element has been transformed
        by the custom sigmoid function.
    """

    return 1 / (1 + torch.exp(-k * x))


class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()
        self.epsilon = 1e-8

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + self.epsilon)


class VGGLoss(pl.LightningModule):
    def __init__(self, vgg_layers=18):
        super().__init__()
        self.vgg = vgg19(weights="VGG19_Weights.DEFAULT").features[:vgg_layers].eval()
        self.loss = nn.MSELoss()
        # self.loss = nn.L1Loss()

        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, gen, high_res):
        # Need to reshape from [B,1,H,W] to [B,3,H,W]
        gen = gen.repeat(1, 3, 1, 1)
        high_res = high_res.repeat(1, 3, 1, 1)

        self.vgg.eval()
        vgg_gen_features = self.vgg(gen)
        vgg_highres_features = self.vgg(high_res)

        # normalise loss by 1/3 as had to expand to 3 channels
        return self.loss(vgg_gen_features, vgg_highres_features) / 3
