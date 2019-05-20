"""Just another implementation on LayoutGAN.
Use TensorFlow, TensorFlow-Graphics, and Keras

TODO list:
- [x] Create the generator.
- [x] Create the relational discriminator.
- [ ] Set up training method.
    - [ ] Set up the loss functions.
    - [ ] Initialize the layouts, feed to the generator.
    - [ ] Train the discriminator.
    - [ ] Train the generator.
- [ ] Run one epoch of training to test the code.

Copyright ©2019-current, Junru Zhong, all rights reserved.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np

# Modify the imports for IDE code completion.
# Refer to https://github.com/tensorflow/tensorflow/issues/26813
from tensorflow.python.keras import datasets, layers, models, activations, optimizers
from tensorflow.python.keras import backend as K


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
    """For MNIST dataset only. Transfer pixel values to geometry points with classes."""
    if greyscale >= thresh:
        return 1
    else:
        return 0


class RelationModule(layers.Layer):
    """Compute the relations between the elements.
    The module here has a same shape of inputs and outputs.
    """

    def __init__(self, num_classes, num_geometry_parameters, num_elements):
        """Layer variables
        W_r: weight matrix creates linear embeddings and context residual information.
        N: the number of elements.
        W_psi: a representation of features of element i.
        W_phi: a representation of features of element j.
        """
        self.num_classes = num_classes
        self.num_geometry_parameters = num_geometry_parameters
        self.num_elements = num_elements
        self.feature_size = self.num_classes + self.num_geometry_parameters

        self.w_r = K.placeholder(shape=[self.feature_size, self.feature_size])
        self.w_psi = layers.Dense(self.feature_size)
        self.w_phi = layers.Dense(self.feature_size)
        self.unary = layers.Dense(self.feature_size)

        super(RelationModule, self).__init__(dynamic=True)

    def call(self, inputs, **kwargs):
        """Forward computations
        U: an unary function computes a representation of the embedded feature for element j.
        H: a dot-product to compute a scalar value on the representations of element i and j.
        Output = W_r\frac{1}{N}\sum_{\forall{j \neq i}}^{} H(f(p_i,\theta_i),f(p_j,\theta_j))U(f(p_j,\theta_j))+f(p_i, \theta_i)
        """
        # This calculation is for a single pair.
        # TODO: Verify and add/not to add the batch loop.
        # TODO: Verify the structure of input tensors.
        f_prime = []
        for idx, i in enumerate(inputs):
            self_attention = K.zeros(i.size)
            for jdx, j in enumerate(inputs):
                if idx == jdx:
                    continue
                else:
                    u = activations.relu(self.unary)
                    dot = layers.dot([self.w_psi, self.w_phi], axes=1)
                    self_attention += dot * u
        f_prime.append(self.w_r * (self_attention / self.num_elements) + i)
        return f_prime

    def compute_output_shape(self, input_shape):
        """Same shape as input"""
        return input_shape


class LayoutGAN:

    def __init__(self):
        self.num_elements = 128
        self.learning_rate = 0.00002
        self.beta1 = 1
        self.beta2 = 1
        self.num_class = 2  # For MNIST
        self.num_geometry_parameter = 2  # For MNIST
        self.feature_size = self.num_class + self.num_geometry_parameter

        # Set up optimizer.
        self.optimizer = optimizers.Adam(
            self.learning_rate, self.beta1, self.beta2)

        # Build the discriminators.
        self.relational_discriminator = self.build_relational_discriminator()
        self.relational_discriminator.compile(
            optimizer=self.optimizer, loss='mean_squared_error')
        # self.wireframe_discriminator = self.build_wireframe_discriminator()

        # Build the generator.
        self.generator = self.build_generator()

    def build_generator(self):

        # The generator
        generator = models.Sequential()

        # Encoder fully connected layers.
        generator.add(layers.Dense(self.feature_size * 2,
                                   input_shape=(self.feature_size,)))
        generator.add(layers.BatchNormalization())
        generator.add(layers.Dense(self.feature_size * 2 * 2))
        generator.add(layers.BatchNormalization())
        # Relation module.
        # TODO: Stack four relation modules when succeeded for one.
        generator.add(layers.Dense(self.feature_size * 2 * 2))
        generator.add(RelationModule(self.num_class,
                                     self.num_geometry_parameter, self.num_elements))
        # Decoder fully connected layers.
        generator.add(layers.Dense(self.feature_size * 2))
        generator.add(layers.BatchNormalization())
        generator.add(layers.Dense(self.feature_size))
        # Branches
        generator.add(layers.Dense(self.num_class))
        generator.add(layers.Dense(self.num_geometry_parameter))

        print(generator.summary())

        return models.Model()

    def build_relational_discriminator(self):

        # The relational discriminator
        relational_discriminator = models.Sequential()

        # Encoder fully connected layers.
        relational_discriminator.add(layers.Dense(
            self.feature_size * 2, input_shape=(self.feature_size,)))
        relational_discriminator.add(layers.BatchNormalization())
        relational_discriminator.add(layers.Dense(self.feature_size * 2 * 2))
        relational_discriminator.add(layers.BatchNormalization())
        # Relation module.
        # TODO: Stack four modules later.
        relational_discriminator.add(layers.Dense(self.feature_size * 2 * 2))
        relational_discriminator.add(RelationModule(
            self.num_class, self.num_geometry_parameter, self.num_elements))
        # Decoder fully connected layers.
        relational_discriminator.add(layers.Dense(self.feature_size * 2))
        relational_discriminator.add(layers.BatchNormalization())
        relational_discriminator.add(layers.Dense(self.feature_size))
        # Branches
        relational_discriminator.add(layers.Dense(self.num_class))
        relational_discriminator.add(layers.Dense(self.num_geometry_parameter))
        # Max pooling
        # relational_discriminator.add(layers.MaxPool1D(self.num_elements))
        # Logits
        # relational_discriminator.add(layers.Dense(self.feature_size, activations='logits'))

        print(relational_discriminator.summary())

        return models.Model()

    def build_wireframe_discriminator(self):
        pass

    def train_by_relational_discriminator(self, epochs, batch_size=128):

        # Load MNIST dataset.
        (train_images, _), (_, _) = datasets.mnist.load_data()
        train_images = train_images.reshape((60000, 28, 28, 1))
        vfunction = np.vectorize(transfer_greyscale_class)
        layout_vectors = vfunction(train_images, 200)

        # TODO: Set up placeholders for adversarial ground truth and fake result.
        # Ground truth: the datasets; fake layouts: initialized according to the article.
        valid = np.ones((batch_size, self.feature_size))
        fake = np.zeros((batch_size, self.feature_size))

        for epoch in range(epochs):

            # Train discriminator

            # Select a random batch of new images.
            idx = np.random.randint(0, layout_vectors[0], batch_size)
            imgs = layout_vectors[idx]

            # TODO: Create the noise here (initialized by some distributions).
            # Class probabilities are one-hot vector and randomly initialized.
            # Geometry parameters are also randomly initialized.
            # Stack these two vectors together.
            # noise_class_probabilities = np.random.???
            # noise_geometry_parameters = np.random.???

            pass


if __name__ == '__main__':
    # Test network
    layoutgan = LayoutGAN()
