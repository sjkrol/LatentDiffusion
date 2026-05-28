
from typing import Tuple

import torch
import torch.nn as nn

import lpips
from models.discriminator import Discriminator

def hinge_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    """
    Hinge loss function for the discriminator. 
    This loss is used to train the discriminator to distinguish between real and fake images.
    @author: Stephen Krol

    :param logits_real: Output of the discriminator for real images (shape: (batch_size, 1))
    :type logits_real: torch.Tensor
    :param logits_fake: Output of the discriminator for generated images (shape: (batch_size, 1))
    :type logits_fake: torch.Tensor

    :return: Hinge loss value
    :rtype: torch.Tensor
    """

    return 0.5 * (torch.mean(torch.relu(1 - logits_real)) + torch.mean(torch.relu(1 + logits_fake)))

class PerceptualLoss(nn.Module):
    """
    PerceptualLoss class defines the perceptual loss function based on the work in 
    the LDM paper. This loss is used to train both a KL 
    """

    def __init__(self, 
                 discriminator_start_step: int,
                 logvar_init: float = 0.0, 
                 kl_weight: float = 1.0, 
                 discriminator_nlayers: int = 3,
                 discriminator_base_channels: int = 64, 
                 discriminator_in_channels: int = 3,
                 discriminator_factor: float = 1.0,
                 discriminator_weight: float = 1.0,
                 perceptual_weight: float = 1.0,
                 device: str = 'cpu',
                 ) -> None:
        
        """
        PerceptualLoss class defines the perceptual loss function based on the work in the LDM paper. 
        This loss is used to train both a KL Variational Autoencoder and a discriminator in an adversarial manner.
        @author: Stephen Krol

        :param discriminator_start_step: Step at which to start applying the discriminator loss (default: 0)
        :type discriminator_start_step: int
        :param logvar_init: Initial value for the log variance of the KL divergence (default: 0.0)
        :type logvar_init: float
        :param kl_weight: Weight for the KL divergence loss (default: 1.0)
        :type kl_weight: float
        :param discriminator_nlayers: Number of layers in the discriminator (default: 3)
        :type discriminator_nlayers: int
        :param discriminator_base_channels: Number of base channels in the discriminator (default: 64)
        :type discriminator_base_channels: int
        :param discriminator_in_channels: Number of input channels for the discriminator (default: 3 for RGB images)
        :type discriminator_in_channels: int
        :param discriminator_factor: Factor to scale the discriminator loss (default: 1.0)
        :type discriminator_factor: float
        :param discriminator_weight: Weight for the discriminator loss (default: 1.0)
        :type discriminator_weight: float
        :param perceptual_weight: Weight for the perceptual loss (default: 1.0)
        :type perceptual_weight: float
        :param device: Device to run the loss computations on (default: 'cpu')
        :type device: str
        """

        super().__init__()

        # define KL divergence weight
        self.kl_weight = kl_weight

        # define perceptual loss using the lpips library
        self.perceptual_loss = lpips.LPIPS(net='vgg').to(device)
        self.perceptual_weight = perceptual_weight

        # set log variance as a learnable parameter
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        # define discriminator and related parameters
        self.discriminator = Discriminator(
            in_channels=discriminator_in_channels,
            base_channels=discriminator_base_channels,
            n_layers=discriminator_nlayers
        ).to(device)

        self.discriminator_start_step = discriminator_start_step
        self.discriminator_loss_fn = hinge_loss
        self.discriminator_factor = discriminator_factor
        self.discriminator_weight = discriminator_weight
    
    def calculate_adaptive_weight(self, 
                                  nll_loss: torch.Tensor, 
                                  g_loss: torch.Tensor, 
                                  last_layer: torch.nn.Module) -> torch.Tensor:
        """
        Calculate adaptive weight for the discriminator loss based on the gradients of the KL divergence and generator losses.
        This is used to balance the training of the generator and discriminator in an adversarial manner.
    
        @author: Stephen Krol

        :param nll_loss: Negative log likelihood loss (KL divergence) for the generator
        :type nll_loss: torch.Tensor
        :param g_loss: Generator loss (discriminator loss for generated images)
        :type g_loss: torch.Tensor
        :param last_layer: The last layer of the generator network, used to compute gradients for adaptive weighting
        :type last_layer: torch.nn.Module

        :return: Adaptive weight for the discriminator loss
        :rtype: torch.Tensor
        """

        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight
    
    def forward(self, 
                inputs: torch.Tensor, 
                reconstructions: torch.Tensor, 
                posteriors: torch.Tensor, 
                optimizer_idx: int, 
                global_step: int, 
                last_layer: torch.nn.Module) -> Tuple[torch.Tensor, dict]:
        """
        Forward method defines the forward pass of the perceptual loss computation. 
        This includes calculating the KL divergence loss, perceptual loss, and discriminator loss (if applicable) 
        for both the generator and discriminator updates.
        @author: Stephen Krol

        :param inputs: Original input images (shape: (batch_size, in_channels, height, width))
        :type inputs: torch.Tensor
        :param reconstructions: Reconstructed images from the generator (shape: (batch_size, in_channels, height, width))
        :type reconstructions: torch.Tensor
        :param posteriors: Posterior distributions from the encoder (used for KL divergence)
        :type posteriors: torch.Tensor
        :param optimizer_idx: Index of the optimizer (0 for generator update, 1 for discriminator update)
        :type optimizer_idx: int
        :param global_step: Current global training step (used to determine when to start applying discriminator loss)
        :type global_step: int
        :param last_layer: The last layer of the generator network, used to compute gradients for adaptive weighting
        :type last_layer: torch.nn.Module

        :return: Tuple containing the total loss for the current update and a dictionary of log values for monitoring training progress
        :rtype: Tuple[torch.Tensor, dict]
        """

        # calculate reconstruction loss (L1) and perceptual loss
        reconstruction_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        perceptual_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
        total_reconstruction_loss = reconstruction_loss + self.perceptual_weight * perceptual_loss

        # calculate negative log likelihood loss (KL divergence) for the KLVAE due to learned log variance
        nll_loss = total_reconstruction_loss / torch.exp(self.logvar) + self.logvar
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]

        # calculate KL divergence loss
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        # VAE update
        if optimizer_idx == 0:

            logits_fake = self.discriminator(reconstructions.contiguous())
            g_loss = -torch.mean(logits_fake) # generator wants to maximize discriminator output for generated images

            if self.discriminator_factor > 0.0 and global_step >= self.discriminator_start_step:
                d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
            else:
                d_weight = 0.0
            
            loss = nll_loss + self.kl_weight * kl_loss + d_weight * self.discriminator_factor * g_loss

            # logs for monitoring training progress
            log = {
                "total_loss": loss.clone().detach().mean(),
                "logvar": self.logvar.detach(),
                "kl_loss": kl_loss.detach().mean(),
                "nll_loss": nll_loss.detach().mean(),
                "reconstruction_loss": reconstruction_loss.detach().mean(),
                "perceptual_loss": perceptual_loss.detach().mean(),
                "d_weight": d_weight.detach(),
                "disc_factor": torch.tensor(self.discriminator_factor),
                "g_loss": g_loss.detach().mean(),
            }

            return loss, log
        
        # discriminator update
        if optimizer_idx == 1:
            
            logits_real = self.discriminator(inputs.contiguous().detach())
            logits_fake = self.discriminator(reconstructions.contiguous().detach())

            # do not apply discriminator loss until a certain number of steps have passed to allow the generator to learn something reasonable
            if global_step >= self.discriminator_start_step:
                self.discriminator_factor = 0

            d_loss = self.discriminator_factor * self.discriminator_loss_fn(logits_real, logits_fake)

            log = {
                "disc_loss": d_loss.clone().detach().mean(),
                "logits_real_mean": logits_real.mean().detach(),
                "logits_fake_mean": logits_fake.mean().detach(),
            }

            return d_loss, log