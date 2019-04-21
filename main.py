from __future__ import print_function
import argparse
import os
import random
import torch

# Root directory for dataset.
dataroot = "data/mnist"  # Currently consider only MNIST dataset.

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 128

# Number of training epochs
num_epochs = 5  # Not provided in the article.

# Leaning rate for optimizers
learning_rate = 0.00002

# Beta1 hyperparameter for Adam optimizers (check the theory).
beta1 = 0.5  # Not provided in the article.

# Number of GPUs available. Use 0 for CPU mode.
n_gpu = 1


# Handle dataset.

