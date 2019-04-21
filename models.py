"""A verification to the idea of LayoutGAN
Referred to https://github.com/sngjuk/LayoutGAN

Implementation of the models.
Copyright ©2019-current, Junru Zhong, All rights reserved.
"""

import torch.nn as nn


class Generator(nn.Module):
    """The generator (in GAN)"""

    def __init__(self, n_gpu, feature_size, class_num, element_num):
        super(Generator, self).__init__()
        self.n_gpu = n_gpu
        self.main = nn.Sequential(
            # Encoder: two fully connected layers, input layout Z.
            nn.Linear(feature_size, feature_size * 2),  # Guessing? Why is a doubled size?
            nn.BatchNorm1d(element_num),
            nn.Linear(feature_size * 2, feature_size * 2 * 2),
            nn.BatchNorm1d(element_num),
            nn.Linear(feature_size * 2 * 2, feature_size * 2 * 2)
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
