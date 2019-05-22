"""Just another implementation on LayoutGAN.
Use TensorFlow, TensorFlow-Graphics, and Keras

TODO list:
- [x] Create the generator.
- [x] Create the relational discriminator.
- [x] Set up training method.
    - [x] Set up the loss functions.
    - [x] Initialize the layouts, feed to the generator.
    - [x] Train the discriminator.
    - [x] Train the generator.
- [ ] Run one epoch of training to test the code.

Copyright ©2019-current, Junru Zhong, all rights reserved.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf
# Modify the imports for IDE code completion.
# Refer to https://github.com/tensorflow/tensorflow/issues/26813
from tensorflow.python.keras import Input, activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import datasets, layers, models, optimizers

# Enable eager execution. Not necessary for TensorFlow 2.0+.
tf.enable_eager_execution()

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


# class RelationModule(layers.Layer):
#     """Compute the relations between the elements.

#     The module here has a same shape of inputs and outputs.
#     W_r: weight matrix creates linear embeddings and context residual information.
#     N: the number of elements.
#     W_psi: a representation of features of element i.
#     W_phi: a representation of features of element j.
#     U: an unary function computes a representation of the embedded feature for element j.
#     H: a dot-product to compute a scalar value on the representations of element i and j.

#     Output = W_r\frac{1}{N}\sum_{\forall{j \neq i}}^{} H(f(p_i,\theta_i),f(p_j,\theta_j))U(f(p_j,\theta_j))+f(p_i, \theta_i)
#     """

#     def __init__(self, num_classes, num_geometry_parameters, num_elements):
#         """Layer variables
#         W_r: weight matrix creates linear embeddings and context residual information.
#         N: the number of elements.
#         W_psi: a representation of features of element i.
#         W_phi: a representation of features of element j.
#         """
#         self.num_classes = num_classes
#         self.num_geometry_parameters = num_geometry_parameters
#         self.num_elements = num_elements
#         self.feature_size = self.num_classes + self.num_geometry_parameters

#         self.w_r = K.placeholder(shape=[self.feature_size, self.feature_size])
#         # self.w_psi = layers.Dense(self.feature_size)
#         # self.w_phi = layers.Dense(self.feature_size)
#         self.unary = layers.Dense(self.feature_size, activation='relu')

#         super(RelationModule, self).__init__(dynamic=True)

#     def build(self, input_shape):
#         # self.w_r = self.add_weight(name='w_r', shape='')
#         return super(RelationModule).build(input_shape)

#     def call(self, inputs, **kwargs):
#         """Forward computations
#         U: an unary function computes a representation of the embedded feature for element j.
#         H: a dot-product to compute a scalar value on the representations of element i and j.
#         Output = W_r\frac{1}{N}\sum_{\forall{j \neq i}}^{} H(f(p_i,\theta_i),f(p_j,\theta_j))U(f(p_j,\theta_j))+f(p_i, \theta_i)
#         """
#         # This calculation is for a single pair.
#         f_prime = []
#         for idx, i in enumerate(inputs):
#             # FIXME: 'tensorflow.python.framework.ops.EagerTensor' object has no attribute 'size'
#             self_attention = K.zeros(i.numpy().size)
#             for jdx, j in enumerate(inputs):
#                 if idx == jdx:
#                     continue
#                 else:
#                     u = self.unary
#                     # TODO: w_psi & w_phi, linear embeddings for f_i and f_j.
#                     w_psi = layers.Dense(self.feature_size)(i)
#                     w_phi = layers.Dense(self.feature_size)(j)
#                     dot = K.dot(w_psi, w_phi)
#                     self_attention += dot * u
#         f_prime.append(self.w_r * (self_attention / self.num_elements) + i)
#         return f_prime

#     def compute_output_shape(self, input_shape):
#         """Same shape as input"""
#         return input_shape


class LayoutGAN:

    def __init__(self):
        self.num_elements = 128
        self.learning_rate = 0.00002
        self.beta1 = 1
        self.beta2 = 1
        self.num_class = 1  # For MNIST, white only.
        self.num_geometry_parameter = 2  # For MNIST
        self.feature_size = self.num_class + self.num_geometry_parameter

        # Set up optimizer.
        self.optimizer = optimizers.Adam(
            self.learning_rate, self.beta1, self.beta2)

        # Build the discriminators.
        self.discriminator = self.build_relational_discriminator()
        self.discriminator.compile(
            optimizer=self.optimizer, loss='mean_squared_error')
        # self.discriminator = self.build_wireframe_discriminator()

        # Build the generator.
        self.generator = self.build_generator()

        # The generator takes noise as input and generates layouts.
        z = Input(shape=(self.feature_size,))
        gen_layouts = self.generator(z)

        # The discriminator takes the generated layouts as input and determines validity.
        validity = self.discriminator(gen_layouts)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = models.Model(z, validity)
        self.combined.compile(optimizer=self.optimizer,
                              loss='mean_squared_error')

    def relation_module(self, inputs):
        """Relation module creates the residual of two elements.
        The module here has a same shape of inputs and outputs.

        inputs: encoded feature.

        W_r: weight matrix creates linear embeddings and context residual information.
        N: the number of elements.
        W_psi: a representation of features of element i.
        W_phi: a representation of features of element j.
        U: an unary function computes a representation of the embedded feature for element j.
        H: a dot-product to compute a scalar value on the representations of element i and j.

        Output = W_r\frac{1}{N}\sum_{\forall{j \neq i}}^{} H(f(p_i,\theta_i),f(p_j,\theta_j))U(f(p_j,\theta_j))+f(p_i, \theta_i)
        """
        w_r = np.random.random_sample(1)
        psi = np.random.random_sample(1)
        phi = np.random.random_sample(1)
        f_prime = []
        # FIXME: Enable dynamic mode to iterate tensors.
        for idx, i in enumerate(inputs):
            self_attention = np.zeros(i.size())
            for jdx, j in enumerate(inputs):
                if idx == jdx:
                    continue
                u = layers.Dense(
                    feature_size * 2 * 2, input_shape=feature_size * 2 * 2, activation='relu')(j)
                i_reshape = np.reshape(i.size(), 1)
                j_reshape = np.reshape(j.size(), 1)
                dot = layers.dot([(i_reshape * psi).t(), j_reshape * phi])
                self_attention += dot * u
            f_prime.append(w_r * (self_attention / num_elements) + i)
        return f_prime

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
        # generator.add(RelationModule(self.num_class, self.num_geometry_parameter, self.num_elements))
        generator.add(layers.Lambda(self.relation_module,
                                    output_shape=(self.feature_size * 2 * 2,)))
        # Decoder fully connected layers.
        generator.add(layers.Dense(self.feature_size * 2))
        generator.add(layers.BatchNormalization())
        generator.add(layers.Dense(self.feature_size))
        # Branches
        generator.add(layers.Dense(self.num_class))
        generator.add(layers.Dense(self.num_geometry_parameter))

        print(generator.summary())

        noise = Input(shape=(self.feature_size,))
        layout = generator(noise)

        return models.Model(noise, layout)

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
        # relational_discriminator.add(RelationModule(self.num_class, self.num_geometry_parameter, self.num_elements))
        relational_discriminator.add(layers.Lambda(
            self.relation_module, output_shape=(self.feature_size * 2 * 2,)))
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

        layout = Input(shape=(self.feature_size,))
        validity = relational_discriminator(layout)

        return models.Model(layout, validity)

    def build_wireframe_discriminator(self):
        pass

    def train_by_relational_discriminator(self, epochs, batch_size=128):

        # Load MNIST dataset.
        # TODO: Change to only distingush one class only.
        (train_images, _), (_, _) = datasets.mnist.load_data()
        # train_images = train_images.reshape((60000, 28, 28, 1))
        vfunction = np.vectorize(transfer_greyscale_class)
        layout_vectors = vfunction(train_images, 200)

        # Set up placeholders for adversarial ground truth and fake result.
        # Ground truth: the datasets; fake layouts: initialized according to the article.
        valid = np.ones((batch_size, self.feature_size))
        fake = np.zeros((batch_size, self.feature_size))

        for epoch in range(epochs):

            # --------------------
            # Train discriminator
            # --------------------

            # Select a random batch of new images.
            # FIXME: 60000 is the assumed maximum index value of the random batch.
            idx = np.random.randint(0, 60000, batch_size)
            real_layouts = layout_vectors[idx]

            # Create the noise here (initialized by some distributions).
            # Class probabilities are randomly initialized.
            # In MNIST case, there is only one class and all of the p_i are 1.
            noise_class_probabilities = np.ones((batch_size, self.num_class))
            # Geometry parameters are also randomly initialized by normal distribution.
            noise_geometry_parameters = np.random.normal(
                0, 1, size=(batch_size, self.num_geometry_parameter))
            # Stack these two vectors together, feed to a Keras variable.
            noise = np.concatenate(
                (noise_class_probabilities, noise_geometry_parameters), axis=1)

            # Generate a batch of new images
            gen_layouts = self.generator.predict(noise)

            # Do training of the discriminator
            d_loss_real = self.discriminator.train_on_batch(
                real_layouts, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_layouts, fake)
            d_loss = np.add(d_loss_real, d_loss_fake)

            # --------------------
            # Train generator
            # --------------------

            # Generate noise.
            noise_class_probabilities = np.ones((batch_size, self.num_class))
            noise_geometry_parameters = np.random.normal(
                0, 1, size=(batch_size, self.num_geometry_parameter))
            noise = np.concatenate(
                (noise_class_probabilities, noise_geometry_parameters), axis=1)

            # Do training of the generator.
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress,
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" %
                  (epoch, d_loss[0], 100*d_loss[1], g_loss))


if __name__ == '__main__':
    # Test network
    layoutgan = LayoutGAN()
    layoutgan.train_by_relational_discriminator(epochs=1, batch_size=128)
