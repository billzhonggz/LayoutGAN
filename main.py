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
import torch.utils.data
import torchvision.datasets

# Root directory for dataset.
dataroot = "data"

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 128

# Number of training epochs
num_epochs = 40  # Not provided in the article.

# Leaning rate for optimizers
learning_rate = 0.00002

# Beta1 hyperparameter for Adam optimizers (check the theory).
beta1 = 0.5  # Not provided in the article.

# Number of GPUs available. Use 0 for CPU mode.
n_gpu = 1


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


def train_mnist():
    # Download MNIST dataset
    _ = torchvision.datasets.MNIST(root=dataroot, train=True, download=True, transform=None)

    # Load MNIST dataset with layout processed.
    train_mnist_layout = MnistLayoutDataset(dataroot)
    train_mnist_layout_loader = torch.utils.data.DataLoader(train_mnist_layout, batch_size=batch_size)

    # TODO: train discriminator with real images.
    # TODO: randomly initialize layout.


if __name__ == '__main__':
    train_mnist()
