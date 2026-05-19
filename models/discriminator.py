
import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    Discriminator class based on the work by https://github.com/CompVis/taming-transformers/tree/master
    taming-transformers.
    @author: Stephen Krol
    """

    def __init__(self, in_channels=3, base_channels=64, n_layers=3):
        super(Discriminator, self).__init__()

        self.norm = nn.BatchNorm2d

        layers = []

        # intial convolutional layer
        layers.append(nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        filter_multiplier = 1
        prev_filter_multiplier = 1

        # downsampling layers
        for i in range(1, n_layers + 1):
            prev_filter_multiplier = filter_multiplier
            filter_multiplier = min(2 ** i, 8)

            if i == n_layers:
                stride = 1
            else:
                stride = 2

            layers.append(nn.Conv2d(base_channels * prev_filter_multiplier, base_channels * filter_multiplier, kernel_size=4, stride=stride, padding=1))
            layers.append(self.norm(base_channels * filter_multiplier))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # final convolutional layer to output a single scalar value
        layers.append(nn.Conv2d(base_channels * filter_multiplier, 1, kernel_size=4, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method defines the forward pass of the discriminator.
        @author: Stephen Krol

        :param x: Input tensor of shape (batch_size, in_channels, height, width)
        :type x: torch.Tensor

        :return: Output tensor of shape (batch_size, 1)
        :rtype: torch.Tensor
        """

        return self.model(x)

def weights_init(m):
    """
    Function from original taming-transformers codebase to initialize the weights of the discriminator.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


if __name__ == "__main__":


    x = torch.randn(1, 3, 256, 256)

    discriminator = Discriminator(in_channels=3, base_channels=64, n_layers=3)
    output = discriminator(x)
    print(output.shape)