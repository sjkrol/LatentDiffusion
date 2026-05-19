
import yaml
import torch
import torch.nn as nn

from utils import ResBlock, SelfAttention, GroupNorm, nonlinearity


class Decoder(nn.Module):
    def __init__(self, final_channels=3, z_channels=4, base_channels=128, in_resolution=256, channel_multipliers=[1, 2, 4, 4], num_res_blocks=2, attention_resolutions=[]):

        super(Decoder, self).__init__()

        # initial convolutional layer to process latent representation to base channels
        self.input_conv = nn.Conv2d(z_channels, channel_multipliers[-1] * base_channels, kernel_size=3, stride=1, padding=1)

        # middle blocks
        self.midblocks = nn.ModuleList()
        self.midblocks.append(ResBlock(channel_multipliers[-1] * base_channels, channel_multipliers[-1] * base_channels, num_groups=32))
        self.midblocks.append(SelfAttention(channel_multipliers[-1] * base_channels))
        self.midblocks.append(ResBlock(channel_multipliers[-1] * base_channels, channel_multipliers[-1] * base_channels, num_groups=32))

        # upsampling
        self.upblocks = nn.ModuleList()
        in_channels = channel_multipliers[-1] * base_channels
        current_resolution = in_resolution // (2 ** (len(channel_multipliers) - 1))

        for i, mult in reversed(list(enumerate(channel_multipliers))):

            out_channels = base_channels * mult

            # pass through residual blocks
            for _ in range(num_res_blocks):
                self.upblocks.append(ResBlock(in_channels, out_channels, num_groups=32))
                in_channels = out_channels

                # attention block if current resolution is in attention_resolutions
                if current_resolution in attention_resolutions:
                    self.upblocks.append(SelfAttention(out_channels))

            # upsample by factor of 2 if not at the end of channel multipliers
            if i != 0:
                self.upblocks.append(nn.Upsample(scale_factor=2, mode='nearest'))
                self.upblocks.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
                current_resolution *= 2
        
        # final output layers
        self.output_norm = GroupNorm(num_groups=32, in_channels=out_channels)
        self.output_conv = nn.Conv2d(out_channels, final_channels, kernel_size=3, stride=1, padding=1)
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method defines the forward pass of the decoder.
        @author: Stephen Krol

        :param x: Input tensor of shape (batch_size, z_channels, height // 8, width // 8)
        :type x: torch.Tensor

        :return: Output tensor of shape (batch_size, final_channels, height, width)
        :rtype: torch.Tensor
        """

        x = self.input_conv(x)

        for block in self.midblocks:
            x = block(x)
        
        for block in self.upblocks:
            x = block(x)
        
        x = nonlinearity(self.output_norm(x))
        return self.output_conv(x)
    
if __name__ == "__main__":

    with open("../config/encoder.yaml", "r") as f:
        config = yaml.safe_load(f)

    x = torch.randn(1, config["model"]["z_channels"], config["model"]["resolution"] // 8, config["model"]["resolution"] // 8)

    decoder = Decoder(
        final_channels=config["model"]["out_channels"],
        z_channels=config["model"]["z_channels"],
        base_channels=config["model"]["base_channels"],
        in_resolution=config["model"]["resolution"],
        channel_multipliers=config["model"]["channel_multipliers"],
        num_res_blocks=config["model"]["num_res_blocks"],
        attention_resolutions=config["model"]["attention_resolutions"]
    )

    x = decoder(x)

    print(x.shape)