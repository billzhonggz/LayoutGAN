"""A verification to the idea of LayoutGAN
Referred to https://github.com/sngjuk/LayoutGAN

Implementation of the models.
Copyright ©2019-current, Junru Zhong, All rights reserved.
"""

import torch.nn as nn


class Generator(nn.Module):
    """The generator (in GAN)"""

    def __init__(self, n_gpu):
        super(Generator, self).__init__()
        self.n_gpu = n_gpu
        self.main = nn.Sequential(
            # Encoder: two fully connected layers, input layout Z.
        )

    def forward(self, *input):
        pass


class Discriminator(nn.Module):
    """The discriminator (in GAN)
    Implement the Wireframe Rendering discriminator.
    """

    def __init__(self, n_gpu):
        super(Discriminator, self).__init__()
        self.n_gpu = n_gpu
        self.main = nn.Sequential(

        )

    def forward(self, *input):
        pass
