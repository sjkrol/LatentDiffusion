

import torch
import yaml
import torch.nn as nn

from .utils import GroupNorm, nonlinearity, ResBlock, SelfAttention

class Encoder(nn.Module):

    def __init__(self, in_channels=3, z_channels=4, base_channels=128, in_resolution=256, num_res_blocks=2, channel_multipliers=[1, 2, 4, 4], attention_resolutions=[]):

        super(Encoder, self).__init__()

        # initial convolutional layer to process input image to base channels
        self.input_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1)

        # instance variables
        self.current_resolution = in_resolution
        self.in_channels = base_channels
        self.attention_resolutions = attention_resolutions

        # downsampling
        self.downblocks = nn.ModuleList()

        for i, mult in enumerate(channel_multipliers):
            out_channels = base_channels * mult

            # pass through residual blocks
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(self.in_channels, out_channels, num_groups=32))
                self.in_channels = out_channels

                # add self-attention block if current resolution is in attention_resolutions
                if self.current_resolution in self.attention_resolutions:
                    self.downblocks.append(SelfAttention(self.in_channels))

            # downsample by factor of 2 if not at the end of channel multipliers
            if i != len(channel_multipliers) - 1:
                self.downblocks.append(nn.Conv2d(self.in_channels, out_channels, kernel_size=3, stride=2, padding=1))
                self.in_channels = out_channels
                self.current_resolution //= 2
        
        # mid blocks
        self.midblocks = nn.ModuleList()
        self.midblocks.append(ResBlock(self.in_channels, self.in_channels, num_groups=32))
        self.midblocks.append(SelfAttention(self.in_channels))
        self.midblocks.append(ResBlock(self.in_channels, self.in_channels, num_groups=32))

        # final output layers
        self.output_norm = GroupNorm(num_groups=32, in_channels=self.in_channels)
        self.output_conv = nn.Conv2d(self.in_channels, z_channels * 2, kernel_size=3, stride=1, padding=1)


    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method defines the forward pass of the encoder.
        @author: Stephen Krol

        :param x: Input tensor of shape (batch_size, in_channels, height, width)
        :type x: torch.Tensor

        :return: Output tensor of shape (batch_size, z_channels * 2, height // 8, width // 8)
        :rtype: torch.Tensor
        """

        x = self.input_conv(x)

        for block in self.downblocks:
            x = block(x)

        for block in self.midblocks:
            x = block(x)

        x = nonlinearity(self.output_norm(x))
        return self.output_conv(x)



if __name__ == "__main__":


    with open("../config/encoder.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    x = torch.randn(1, config["model"]["out_channels"], config["model"]["resolution"], config["model"]["resolution"])
    print(x.shape)
    
    
    encoder = Encoder(config["model"]["out_channels"],
                      config["model"]["z_channels"],
                      config["model"]["base_channels"],
                      config["model"]["resolution"],
                      config["model"]["num_res_blocks"],
                      config["model"]["channel_multipliers"],
                      config["model"]["attention_resolutions"])

    x = encoder(x)
    print(x.shape)