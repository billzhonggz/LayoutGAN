from __future__ import print_function
import argparse
import os
import random
import torch


# Root directory for dataset.
dataroot = "data/mnist"

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 128

