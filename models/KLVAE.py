

import yaml
import torch
import torch.nn as nn

from decoder import Decoder
from encoder import Encoder
from utils import DiagonalGuassianDistribution

class KLVAE(nn.Module):

    def __init__(self,
                 embed_dim=4, 
                 in_channels=3, 
                 z_channels=4, 
                 base_channels=128, 
                 in_resolution=256, 
                 num_res_blocks=2, 
                 channel_multipliers=[1, 2, 4, 4], 
                 attention_resolutions=[]):
        """
        KLVAE class defines the overall architecture of the KL Variational Autoencoder.
        @author: Stephen Krol

        :param embed_dim: Dimension of the embedding (default: 4)
        :type embed_dim: int
        :param in_channels: Number of input channels (default: 3 for RGB images)
        :type in_channels: int
        :param z_channels: Number of channels in the latent representation (default: 4)
        :type z_channels: int
        :param base_channels: Number of base channels for the encoder and decoder (default:128)
        :type base_channels: int
        :param in_resolution: Resolution of the input images (default: 256)
        :type in_resolution: int
        :param num_res_blocks: Number of residual blocks in the encoder and decoder (default: 2)
        :type num_res_blocks: int
        :param channel_multipliers: Multipliers for the number of channels at each resolution (default: [1, 2, 4, 4])
        :type channel_multipliers: list
        :param attention_resolutions: Resolutions at which to apply attention (default: [])
        :type attention_resolutions: list
        """
        super(KLVAE, self).__init__()

        self.encoder = Encoder(
            in_channels=in_channels,
            z_channels=z_channels,
            base_channels=base_channels,
            in_resolution=in_resolution,
            num_res_blocks=num_res_blocks,
            channel_multipliers=channel_multipliers,
            attention_resolutions=attention_resolutions
        )

        self.decoder = Decoder(
            final_channels=in_channels,
            z_channels=z_channels,
            base_channels=base_channels,
            in_resolution=in_resolution,
            num_res_blocks=num_res_blocks,
            channel_multipliers=channel_multipliers,
            attention_resolutions=attention_resolutions
        )

        self.quant_conv = nn.Conv2d(z_channels * 2, embed_dim * 2, kernel_size=1)
        self.post_quant_conv = nn.Conv2d(embed_dim, z_channels, kernel_size=1)
        self.embed_dim = embed_dim # if we want to use a different embedding dimension than the number of latent channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method defines the forward pass of the KL Variational Autoencoder.
        @author: Stephen Krol

        :param x: Input tensor of shape (batch_size, in_channels, height, width)
        :type x: torch.Tensor

        :return: Reconstructed tensor of shape (batch_size, in_channels, height, width)
        :rtype: torch.Tensor
        """

        z = self.encode(x)
        z_sample = z.sample()
        x_recon = self.decode(z_sample)
        
        return x_recon

    def encode(self, x: torch.Tensor) -> DiagonalGuassianDistribution:
        """
        Encode method encodes the input image into a latent distribution.
        @author: Stephen Krol

        :param x: Input tensor of shape (batch_size, in_channels, height, width)
        :type x: torch.Tensor

        :return: Latent distribution represented as a DiagonalGaussianDistribution object
        :rtype: DiagonalGuassianDistribution
        """

        h = self.quant_conv(self.encoder(x))
        mean, logvar = torch.chunk(h, 2, dim=1)

        return DiagonalGuassianDistribution(mean, logvar)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode method decodes a latent sample back into the image space.
        @author: Stephen Krol
        
        :param z: Latent sample tensor of shape (batch_size, z_channels, height // 8, width // 8)
        :type z: torch.Tensor

        :return: Reconstructed tensor of shape (batch_size, in_channels, height, width)
        :rtype: torch.Tensor
        """

        return self.decoder(self.post_quant_conv(z))



if __name__ == "__main__":

    with open("../config/encoder.yaml", "r") as f:
        config = yaml.safe_load(f)

    x = torch.randn(1, config["model"]["out_channels"], config["model"]["resolution"], config["model"]["resolution"])

    klvae = KLVAE(
        embed_dim=config["model"]["embed_dim"],
        in_channels=config["model"]["out_channels"],
        z_channels=config["model"]["z_channels"],
        base_channels=config["model"]["base_channels"],
        in_resolution=config["model"]["resolution"],
        num_res_blocks=config["model"]["num_res_blocks"],
        channel_multipliers=config["model"]["channel_multipliers"],
        attention_resolutions=config["model"]["attention_resolutions"]
    )

    x_recon = klvae(x)
    print(x_recon.shape)
    print(x_recon)