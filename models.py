"""A verification to the idea of LayoutGAN
Referred to https://github.com/sngjuk/LayoutGAN

Implementation of the models.
Copyright ©2019-current, Junru Zhong, All rights reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as function

import numpy as np


# Draw shape
def pts(name, ts):
    print(name + ' shape:', np.shape(ts))


# Stacked relation module
def relation_module(out, unary, psi, phi, wr):
    element_num = out.size(1)  # TODO: Inspect the output of the encoded feature.
    batch_res = []
    for bdx, batch in enumerate(out):
        f_prime = []
        # i, j are two elements.
        for idx, i in enumerate(batch):
            self_attention = torch.Tensor(torch.zeros(i.size(0)))
            for jdx, j in enumerate(batch):
                if idx == jdx:
                    continue
                u = function.relu(unary(j))
                iv = i.view(i.size(0), 1)
                jv = j.view(j.size(0), 1)
                dot = (torch.mm((iv * psi).t(), jv * phi)).squeeze()
                self_attention += dot * u
            f_prime.append(wr * (self_attention / element_num) + i)
        batch_res.append(torch.stack(f_prime))
    return torch.stack(batch_res)


class Generator(nn.Module):
    """The generator (in GAN)"""

    def __init__(self, n_gpu, feature_size, class_num, element_num):
        super(Generator, self).__init__()
        self.n_gpu = n_gpu

        # Encoder: two fully connected layers, input layout Z.
        self.encoder_fc1 = nn.Linear(feature_size, feature_size * 2)  # Guessing? Why is a doubled size?
        self.encoder_batch_norm1 = nn.BatchNorm1d(element_num)
        self.encoder_fc2 = nn.Linear(feature_size * 2, feature_size * 2 * 2)
        self.encoder_batch_norm2 = nn.BatchNorm1d(element_num)
        self.encoder_fc3 = nn.Linear(feature_size * 2 * 2, feature_size * 2 * 2)

        # Relation model 1
        self.relation1_unary = nn.Linear(feature_size * 2 * 2,
                                         feature_size * 2 * 2)  # Unary function U, from "Non-local Neural Network"
        self.relation1_psi = torch.FloatTensor(torch.rand(1))  # \psi
        self.relation1_phi = torch.FloatTensor(torch.rand(1))  # \phi
        self.relation1_wr = torch.FloatTensor(torch.rand(1))  # W_r

        # Relation model 2
        self.relation2_unary = nn.Linear(feature_size * 2 * 2,
                                         feature_size * 2 * 2)  # Unary function U, from "Non-local Neural Network"
        self.relation2_psi = torch.FloatTensor(torch.rand(1))  # \psi
        self.relation2_phi = torch.FloatTensor(torch.rand(1))  # \phi
        self.relation2_wr = torch.FloatTensor(torch.rand(1))  # W_r

        # Relation model 3
        self.relation3_unary = nn.Linear(feature_size * 2 * 2,
                                         feature_size * 2 * 2)  # Unary function U, from "Non-local Neural Network"
        self.relation3_psi = torch.FloatTensor(torch.rand(1))  # \psi
        self.relation3_phi = torch.FloatTensor(torch.rand(1))  # \phi
        self.relation3_wr = torch.FloatTensor(torch.rand(1))  # W_r

        # Relation model 4
        self.relation4_unary = nn.Linear(feature_size * 2 * 2,
                                         feature_size * 2 * 2)  # Unary function U, from "Non-local Neural Network"
        self.relation4_psi = torch.FloatTensor(torch.rand(1))  # \psi
        self.relation4_phi = torch.FloatTensor(torch.rand(1))  # \phi
        self.relation4_wr = torch.FloatTensor(torch.rand(1))  # W_r

        # Decoder, two fully connected layers.
        self.decoder_fc1 = nn.Linear(feature_size * 2 * 2, feature_size * 2)
        self.decoder_batch_norm1 = nn.BatchNorm1d(element_num)
        self.decoder_fc2 = nn.Linear(feature_size * 2, feature_size)

        # Branch
        self.branch_fc1 = nn.Linear(feature_size, class_num)
        self.branch_fc2 = nn.Linear(feature_size, feature_size - class_num)

    def forward(self, input):
        # Encoder
        out = function.relu(self.encoder_batch_norm1(self.encoder_fc1(input)))
        out = function.relu(self.encoder_batch_norm2(self.encoder_fc2(out)))
        encoded = function.sigmoid(self.encoder_fc3(out))

        # Stacked relation module
        relation_residual_1 = relation_module(encoded, self.relation1_unary, self.relation1_psi,
                                              self.relation1_phi, self.relation1_wr)
        relation_residual_2 = relation_module(relation_residual_1, self.relation2_unary, self.relation2_psi,
                                              self.relation2_phi, self.relation2_wr)
        relation_residual_3 = relation_module(relation_residual_2, self.relation3_unary, self.relation3_psi,
                                              self.relation3_phi, self.relation3_wr)
        relation_residual_4 = relation_module(relation_residual_3, self.relation4_unary, self.relation4_psi,
                                              self.relation4_phi, self.relation4_wr)

        # Decoder
        out = function.relu(self.decoder_batch_norm1(self.decoder_fc1(relation_residual_4)))
        out = function.relu(self.decoder_fc2(out))

        # Branch
        syn_cls = self.branch_fc1(out)
        syn_geo = self.branch_fc2(out)

        # Synthesized layout
        res = torch.cat((syn_cls, syn_geo), 2)
        pts('res', res)

        return res


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
