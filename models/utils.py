

import torch
import torch.nn as nn

def GroupNorm(num_groups: int, in_channels: int) -> torch.nn.GroupNorm:
    """
    Function normalises output of model using group norm.
    @author: Stephen Krol
    """

    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

def nonlinearity(x: torch.Tensor) -> torch.nn.SiLU:
    """
    Function applies non-linearity to output of model using SiLU activation function.
    @author: Stephen Krol
    """
    
    return torch.nn.functional.silu(x)


class ResBlock(nn.Module):
    """
    Class defines a residual block used in the encoder and decoder of the latent diffusion model.
    @author: Stephen Krol
    """

    def __init__(self, in_channels: int, out_channels: int, num_groups: int):
        """
        Constructor method initializes the layers of the residual block.
        @author: Stephen Krol

        :param in_channels: Number of input channels
        :type in_channels: int
        :param out_channels: Number of output channels
        :type out_channels: int
        :param num_groups: Number of groups for group normalization
        :type num_groups: int
        """

        super(ResBlock, self).__init__()

        self.norm1 = GroupNorm(num_groups=num_groups, in_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.norm2 = GroupNorm(num_groups=num_groups, in_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method defines the forward pass of the residual block.
        @author: Stephen Krol

        :param x: Input tensor of shape (batch_size, in_channels, height, width)
        :type x: torch.Tensor

        :return: Output tensor of shape (batch_size, out_channels, height, width)
        :rtype: torch.Tensor
        """

        h = nonlinearity(self.norm1(x))
        h = self.conv1(h)

        h = nonlinearity(self.norm2(h))
        h = self.conv2(h)

        return h + self.shortcut(x)
    

class SelfAttention(nn.Module):
    """
    Class defines a self-attention block used in the encoder and decoder of the latent diffusion model.
    @author: Stephen Krol
    """
    
    def __init__(self, in_channels: int):
        """
        Constructor method initializes the layers of the vanilla attention block.
        @author: Stephen Krol

        :param in_channels: Number of input channels
        :type in_channels: int
        """

        super(SelfAttention, self).__init__()

        self.Wq = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.Wk = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.Wv = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.Wo = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)

        self.norm = GroupNorm(num_groups=32, in_channels=in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method defines the forward pass of the vanilla attention block.
        @author: Stephen Krol

        :param x: Input tensor of shape (batch_size, in_channels, height, width)
        :type x: torch.Tensor

        :return: Output tensor of shape (batch_size, in_channels, height, width)
        :rtype: torch.Tensor
        """

        B, C, H, W = x.shape

        q = self.Wq(x).view(B, C, H * W).permute(0, 2, 1)
        k = self.Wk(x).view(B, C, H * W)
        v = self.Wv(x).view(B, C, H * W).permute(0, 2, 1)

        attn_weights = torch.bmm(q, k) / (C ** 0.5)
        attn_weights = torch.softmax(attn_weights, dim=-1)

        attn_output = torch.bmm(attn_weights, v).permute(0, 2, 1).view(B, C, H, W)
        attn_output = self.Wo(attn_output)

        return attn_output + x

class DiagonalGuassianDistribution:
    """
    Class defines a diagonal Gaussian distribution used in the reparameterization trick of the latent diffusion model.
    @author: Stephen Krol
    """
    
    def __init__(self, mean: torch.Tensor, logvar: torch.Tensor):
        """
        Constructor method initializes the mean and log variance of the diagonal Gaussian distribution.
        @author: Stephen Krol

        :param mean: Mean of the diagonal Gaussian distribution
        :type mean: torch.Tensor
        :param logvar: Log variance of the diagonal Gaussian distribution
        :type logvar: torch.Tensor
        """

        self.mean = mean
        self.logvar = torch.clamp(logvar, min=-30.0, max=20.0)

    def sample(self) -> torch.Tensor:
        """
        Sample method samples from the diagonal Gaussian distribution using the reparameterization trick.
        @author: Stephen Krol

        :return: Sampled tensor of shape (batch_size, z_channels, height, width)
        :rtype: torch.Tensor
        """

        std = torch.exp(0.5 * self.logvar)
        eps = torch.randn_like(std).to(std.device)

        return self.mean + eps * std
    
    def kl(self) -> torch.Tensor:
        """
        KL method computes the KL divergence between the diagonal Gaussian distribution and a standard normal distribution.
        @author: Stephen Krol

        :return: KL divergence value
        :rtype: torch.Tensor
        """
        
        return 0.5 * torch.sum(self.mean.pow(2) + self.logvar.exp() - self.logvar - 1, dim=[1, 2, 3])



