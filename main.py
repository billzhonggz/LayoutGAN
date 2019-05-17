"""Just another implementation on LayoutGAN.
Use TensorFlow, TensorFlow-Graphics

Copyright ©2019-current, Junru Zhong, all rights reserved.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

from tensorflow.keras import datasets, layers, models


# Enable eager execution. Not necessary for TensorFlow 2.0+.
# tf.enable_eager_execution()

# def load_mnist(thresh=200):
#     """This function loads the MNIST dataset from TensorFlow official Datasets."""
#     # Download from Google
#     mnist_builder = tfds.builder('mnist')
#     mnist_builder.download_and_prepare()
#     # Transfer to TensorFlow `Dataset` object.
#     datasets = mnist_builder.as_dataset()
#     train_dataset = datasets['train']
#     # print(isinstance(train_dataset, tf.data.Dataset))
#     # Transfer MNIST images to tensors with geometry information in the iterator.
#     iterator = tf.compat.v1.data.make_one_shot_iterator(train_dataset)
#     # TODO: Find the API in r2.0 to feed iterable data to the model.
#     # Get one element from the iterator.
#     element = iterator.get_next()['image']
#     # Pick the pixels (as a 2D numpy array).
#     pixels = element.numpy()[:, :, 0]
#     # For each image, the layout vector shape is exactly same as the image.
#     # 0 -> black, 1 -> white.
#     layout_vector = np.zeros((28, 28))
#     for i in range(0, 28):
#         for j in range(0, 28):
#             if pixels[i, j] >= thresh:
#                 layout_vector[i, j] = 1
#             else:
#                 layout_vector[i, j] = 0
#     print(layout_vector)


def transfer_greyscale_class(greyscale, thresh=200):
    if greyscale >= thresh:
        return 1
    else:
        return 0


feature_size = 128  # TODO: change later.


# The relation module.
class RelationModule(layers.Layer):
    """The relation module."""

    def __init__(self):
        super(RelationModule, self).__init__()
        unary = layers.Dense(feature_size * 2 * 2, input_shape=(feature_size * 2 * 2,))
        psi = self.add_weight(shape=())


# The generator
generator = models.Sequential()
generator.add(layers.Dense(feature_size * 2, input_shape=(feature_size,)))  # layers.Dense(output_shape, input_shape=?)
generator.add(layers.Dense(feature_size * 2 * 2))
generator.add(layers.Dense(feature_size * 2 * 2))
generator.add(RelationModule)

if __name__ == '__main__':
    print(tf.keras.__version__)
    # Load training data.
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
    train_images = train_images.reshape((60000, 28, 28, 1))
    vfunction = np.vectorize(transfer_greyscale_class)
    layout_vector = vfunction(train_images, 200)
    # print(layout_vector)
