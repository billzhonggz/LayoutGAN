"""A verification to the idea of LayoutGAN
Referred to https://github.com/sngjuk/LayoutGAN

Entrance of the program.
Copyright ©2019-current, Junru Zhong, All rights reserved.
"""

from __future__ import print_function
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets

import models

# Root directory for dataset.
dataroot = "data"

# Number of workers for dataloader
workers = 2

# Number of GPUs available. Use 0 for CPU mode.
n_gpu = 1

# GPU device
device = torch.device("cuda:0" if (torch.cuda.is_available() and n_gpu > 0) else "cpu")


# Handle dataset. Transform pixel images to point sets.
class MnistLayoutDataset(torch.utils.data.Dataset):
    """MNIST dataset and create torch dataset object."""

    def __init__(self, path, element_num=128, gt_thresh=200):
        super(MnistLayoutDataset, self).__init__()
        self.train_data = torch.load(path + "/MNIST/processed/training.pt")[0]
        self.element_num = element_num
        self.gt_thresh = gt_thresh  # Guess: a threshold (阈值) to indicate this pixel is lighted (0-255).

    def __getitem__(self, index):
        """Extract layout features from images."""
        img = self.train_data[index]  # Load an image.
        gt_values = []

        for id, i in enumerate(img):
            for jd, j in enumerate(i):
                if j >= self.gt_thresh:  # If the current grayscale value is larger than the threshold, note this point.
                    # Create the layout element.
                    gt_values.append(torch.FloatTensor([1, np.float32(2 * id + 1) / 56, np.float32(2 * jd + 1) / 56]))

        graph_elements = []

        # Shuffle, insert the images in a random order.
        for _ in range(self.element_num):
            ridx = random.randint(0, len(gt_values) - 1)
            graph_elements.append(gt_values[ridx])

        # MNIST layout elements format [1, x, y]
        return torch.stack(graph_elements)

    def __len__(self):
        return len(self.train_data)


def real_loss(D_out, smooth=False):
    """Loss function from the discriminator to the generator (when result is real)."""
    labels = None
    batch_size = D_out.size(0)
    if smooth:
        labels = torch.ones(batch_size) * 0.9
    else:
        labels = torch.ones(batch_size)
    crit = nn.BCEWithLogitsLoss()
    loss = crit(D_out.squeeze(), labels)
    return loss


def fake_loss(D_out):
    """Loss function from the discriminator to the generator (when result is fake)."""
    batch_size = D_out.size(0)
    labels = torch.zeros(batch_size)
    crit = nn.BCEWithLogitsLoss()
    loss = crit(D_out.squeeze(), labels)
    return loss


def train_mnist():
    # Batch size during training
    batch_size = 128
    # Number of classes
    cls_num = 1
    # Number of geometry parameter
    geo_num = 2
    # Number of training epochs
    num_epochs = 40  # Not provided in the article.
    # Leaning rate for optimizers
    learning_rate = 0.00002
    # Beta1/2 hyperparameter for Adam optimizers (check the theory).
    beta1 = 1.0  # Not provided in the article.
    beta2 = 1.0

    # Download MNIST dataset
    _ = torchvision.datasets.MNIST(root=dataroot, train=True, download=True, transform=None)

    # Load MNIST dataset with layout processed.
    train_mnist_layout = MnistLayoutDataset(dataroot)
    train_mnist_layout_loader = torch.utils.data.DataLoader(train_mnist_layout, batch_size=batch_size)

    # Initialize the generator and discriminator.
    generator = models.Generator(n_gpu, class_num=1, element_num=128, feature_size=3).to(device)
    discriminator = models.RelationDiscriminator(n_gpu, class_num=1, element_num=128, feature_size=3).to(device)
    print(generator)  # Check information of the generator.
    print(discriminator)  # Check information of the discriminator.

    # Initialize optimizers for models.
    print('Initialize optimizers.')
    generator_optimizer = optim.Adam(generator.parameters(), learning_rate)
    discriminator_optimizer = optim.Adam(discriminator.parameters(), learning_rate)

    # Initialize training parameters.
    print('Initialize traning.')
    generator.train()
    discriminator.train()

    # Start training.
    for epoch in range(num_epochs):
        print('Start to train epoch %d.' % epoch + 1)
        for batch_i, real_images in enumerate(train_mnist_layout_loader):
            batch_size = real_images.size(0)

            # Train discriminator
            discriminator_optimizer.zero_grad()
            discriminator_real = discriminator(real_images)
            discriminator_real_loss = real_loss(discriminator_real, False)

            # TODO: Fix errors in random layout.
            zlist = []
            for i in range(batch_size):
                cls_z = np.ones((batch_size, cls_num))
                geo_z = np.random.normal(0, 1, size=(batch_size, geo_num))

                z = torch.FloatTensor(np.concatenate((cls_z, geo_z), axis=1))
                zlist.append(z)

            fake_images = generator(torch.stack(zlist))

            discriminator_fake = discriminator(fake_images)
            discriminator_fake_loss = fake_loss(discriminator_fake)

            discriminator_loss = discriminator_real_loss + discriminator_fake_loss
            discriminator_loss.backward()
            discriminator_optimizer.step()

            # Train generator
            generator_optimizer.zero_grad()

            # TODO: Fix errors in random layout.
            zlist2 = []
            for i in range(batch_size):
                cls_z = np.ones((batch_size, cls_num))
                geo_z = np.random.normal(0, 1, size=(batch_size, geo_num))

                z = torch.FloatTensor(np.concatenate((cls_z, geo_z), axis=1))
                zlist2.append(z)

            fake_images2 = generator(torch.stack(zlist2))
            discriminator_fake = discriminator(fake_images2)
            generator_loss = real_loss(discriminator_fake, False)

            print('Epoch [{:5d}/{:5d}] | discriminator_loss: {:6.4f} | generator_loss: {:6.4f}'.format(epoch + 1,
                                                                                                       num_epochs,
                                                                                                       discriminator_loss.item(),
                                                                                                       generator_loss.item()))


if __name__ == '__main__':
    train_mnist()
