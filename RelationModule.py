"""Relation module for LayoutGAN.

Copyright ©2019-current, Junru Zhong, all rights reserved.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K


class RelationModule(layers.Layer):
    """Compute the relations between the elements.

    The module here has a same shape of inputs and outputs.
    W_r: weight matrix creates linear embeddings and context residual information.
    N: the number of elements.
    W_psi: a representation of features of element i.
    W_phi: a representation of features of element j.
    U: an unary function computes a representation of the embedded feature for element j.
    H: a dot-product to compute a scalar value on the representations of element i and j.

    Output = W_r\frac{1}{N}\sum_{\forall{j \neq i}}^{} H(f(p_i,\theta_i),f(p_j,\theta_j))U(f(p_j,\theta_j))+f(p_i, \theta_i)
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
        self.w_psi = layers.Embedding(self.feature_size, 1)
        self.w_phi = layers.Embedding(self.feature_size, 1)
        self.unary = layers.Dense(self.feature_size, activation='relu')

        super(RelationModule, self).__init__(dynamic=True)

    # def build(self, input_shape):
    #     # self.w_r = self.add_weight(name='w_r', shape='')
    #     return super(RelationModule).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward computations
        U: an unary function computes a representation of the embedded feature for element j.
        H: a dot-product to compute a scalar value on the representations of element i and j.
        Output = W_r\frac{1}{N}\sum_{\forall{j \neq i}}^{} H(f(p_i,\theta_i),f(p_j,\theta_j))U(f(p_j,\theta_j))+f(p_i, \theta_i)
        """
        # This calculation is for a single pair.
        batch_residual = []
        for _, batch in enumerate(inputs):
            f_prime = []
            for idx, i in enumerate(batch):
                self_attention = K.zeros(i.numpy().size)
                for jdx, j in enumerate(batch):
                    if idx == jdx:
                        continue
                    else:
                        u = self.unary
                        # TODO: How to do linear embeddings.
                        w_psi = layers.Embedding(self.feature_size, 1)(i)
                        w_phi = layers.Embedding(self.feature_size, 1)(j)
                        dot = layers.dot([w_psi, w_phi], 1)
                        self_attention += layers.multiply([dot, u])
                f_prime.append(self.w_r * (self_attention / self.num_elements) + i)
            batch_residual.append(f_prime)
        return batch_residual

    def compute_output_shape(self, input_shape):
        """Same shape as input"""
        return input_shape