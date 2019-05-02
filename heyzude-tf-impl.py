"""Another implementation using TensorFlow
From https://github.com/heyzude/layoutGAN-implementation
"""

import tensorflow as tf
from random import randint
# from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import math
import os


class layoutGAN(object):
    model_name = 'layoutGAN'

    def __init__(self, sess, epoch, p, theta, polygon_type, geopara_range, geopara_mean, geopara_std, geo_range,
                 which_discriminator, mu=0, std=0.02, learning_rate=0.00002, beta1=0.5, beta2=0.999):
        self.sess = sess
        self.epoch = epoch
        # self.batch_size = batch_size
        self.data_p, self.data_theta = p, theta
        self.num_elements = len(self.data_p)
        self.num_classes = len(self.data_p[0])  # assume that each p has dimension of num_classes
        self.num_geopara = len(self.data_theta[0])  # number of geometric parameters ine one theta
        self.geopara_range = geopara_range  # for example, in case of MNIST, the geometric parameter is in form of (x, y) with range of [0, 28] for each x, y. in form of [[geopara1_lower, geopara1_upper,] ... ]
        self.geopara_mean = geopara_mean  # in form of [geopara1_mean, geopara2_mean, ...]
        self.geopara_std = geopara_std  # in form of [geopara1_std, geopara2_std, ...]
        self.polygon_type = polygon_type
        self.geo_range = geo_range  # range of 2-D canvas of our data. in form of [[x_lowerbound, x_upperbound], [y_lowerbound, y_upperbound]]
        self.which_discriminator = which_discriminator
        self.mu = mu
        self.std = std
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2

        self.weight_init = tf.random_normal_initializer(mean=self.mu, stddev=self.std)
        # self.num_batches = self.num_elements // self.batch_size

    def generator(self, reuse=False):
        with tf.variable_scope("generator", reuse=reuse):

            # uniform distribution of class into one-hot form
            rand_classes = [randint(0, self.num_classes - 1) for i in range(self.num_elements)]
            p = tf.one_hot(rand_classes, depth=self.num_classes, dtype=tf.float32)

            # random gaussian distribution of geometric parameters
            theta = np.random.normal(self.geopara_mean[0], self.geopara_std[0], (self.num_elements, 1))
            for i in range(1, self.num_geopara):
                temp_theta = np.random.normal(self.geopara_mean[i], self.geopara_std[i], (self.num_elements, 1))
                theta = np.hstack((theta, temp_theta))

            input_tensor = tf.concat(p, theta, axis=1)

            k = self.num_classes + self.num_geopara

            # encoder multi layers
            g_e_w1 = tf.get_variable('g_e_w1', shape=[k, (k) * 2], initializer=self.weight_init)
            g_e_b1 = tf.get_variable('g_e_b1', shape=[(k) * 2], initializer=self.weight_init)
            g_e_w2 = tf.get_variable('g_e_w2', shape=[(k) * 2, (k) * 2], initializer=self.weight_init)
            g_e_b2 = tf.get_variable('g_e_b2', shape=[(k) * 2], initializer=self.weight_init)
            g_e_w3 = tf.get_variable('g_e_w3', shape=[(k) * 2, k], initializer=self.weight_init)
            g_e_b3 = tf.get_variable('g_e_b3', shape=[k], initializer=self.weight_init)

            # decoder multi layers
            g_d_w1 = tf.get_variable('g_d_w1', shape=[k, (k) * 2], initializer=self.weight_init)
            g_d_b1 = tf.get_variable('g_d_b1', shape=[(k) * 2], initializer=self.weight_init)
            g_d_w2 = tf.get_variable('g_d_w2', shape=[(k) * 2, (k) * 2], initializer=self.weight_init)
            g_d_b2 = tf.get_variable('g_d_b2', shape=[(k) * 2], initializer=self.weight_init)
            g_d_w3 = tf.get_variable('g_d_w3', shape=[(k) * 2, k], initializer=self.weight_init)
            g_d_b3 = tf.get_variable('g_d_b3', shape=[k], initializer=self.weight_init)

            g_d_p_w = tf.get_variable('g_d_p_w', shape=[self.num_classes, self.num_classes],
                                      initializer=self.weight_init)
            g_d_theta_w = tf.get_variable('g_d_theta_w', shape=[self.num_geopara, self.num_geopara],
                                          initializer=self.weight_init)

            # Embedding FC Layer
            L1 = tf.nn.relu(tf.matmul(input_tensor, g_e_w1) + g_e_b1)
            L2 = tf.nn.relu(tf.matmul(L1, g_e_w2) + g_e_b2)
            relation_layer = dict()
            relation_layer[0] = tf.nn.relu(tf.matmul(L2, g_e_w3) + g_e_b3)  # shape of []

            g_sa_hi_w = [tf.get_variable('g_sa_hi_w' + str(i), shape=[k, k], initializer=self.weight_init) for i in
                         range(4)]
            g_sa_hj_w = [tf.get_variable('g_sa_hj_w' + str(i), shape=[k, k], initializer=self.weight_init) for i in
                         range(4)]
            g_sa_u_w = [tf.get_variable('g_sa_u_w' + str(i), shape=[k, k], initializer=self.weight_init) for i in
                        range(4)]
            g_sa_outer_w = [tf.get_variable('g_sa_outer_w' + str(i), shape=[k, k], initializer=self.weight_init) for i
                            in range(4)]
            # g_sa_sigma = [tf.get_variable('g_sa_sigma' + str(i), shape=[1,k], initializer=tf.constant_initializer([[0]*k])) for i in range(4)]

            # Self-Attention + Residual connection Block
            for n in range(3):  # total 4 layer -> forward for 3 steps.
                for i in range(self.num_elements):
                    g_sa_hi = tf.matmul(tf.reshape(relation_layer[n][i], shape=[1, k]), g_sa_hi_w[n])
                    sigma_list = []
                    for j in range(self.num_elements):
                        if i != j:
                            g_sa_hj = tf.matmul(tf.reshape(relation_layer[n][j], shape=[1, k]), g_sa_hj_w[n])
                            H = tf.tensordot(g_sa_hi, g_sa_hj, 1)
                            u = tf.matmul(tf.reshape(relation_layer[n][j], shape=[1, k]), g_sa_u_w[n])
                            sigma_list.append(tf.math.scalar_mul(H, u))

                    sigma = tf.math.add_n(sigma_list)

                    relation_layer[n + 1][i] = tf.reshape(
                        tf.matmul(sigma * float(1. / self.num_elements), g_sa_outer_w[n]) + tf.reshape(
                            relation_layer[n][i], shape=[1, k]),
                        shape=[k])  # residual addtion   # have to fix this with tf.scatter_nd

            # Shape of relation_layer[3] is [self.num_elements, k]
            # decoder part
            L1_ = tf.nn.relu(tf.matmul(relation_layer[3], g_d_w1) + g_d_b1)
            L2_ = tf.nn.relu(tf.matmul(L1_, g_d_w2) + g_d_b2)
            L3_ = tf.nn.relu(tf.matmul(L2_, g_d_w3) + g_d_b3)

            # now, let's split into p FC layer part and theta FC layer part
            split_p, split_theta = tf.split(L3_, [self.num_classes, self.num_geopara], axis=1)
            gened_p = tf.nn.sigmoid(tf.matmul(split_p, g_d_p_w))
            gened_theta = tf.nn.sigmoid(tf.matmul(split_theta, g_d_theta_w))
            # gened_theta = tf.matmul(gened_theta, [math.fabs(range[1]-range[0]) for range in self.geopara_range]) + [min(range[0], range[1]) for range in self.geopara_range]

            # honestly, I am not quite sure about whether applying sigmoid to theta (geometric marameter) part is correct or not.
            # for probability, sigmoid is fine. But when I have to present graphical representation, value of range [0, 1] doesn't quite seem to be right.
            # according to the paper, theta itself can be coordinates, which is not 'sigmoid-representable' type of number.
            # the commented line actually scales and translates the sigmoided one to make things reasonable here (at leat to me)...
            # maybe not doing sigmoid is right? I assume that every theta representation has range.
            # For examle, mnist: [0, 28] for each coordinate.

            # but, if discriminator gets normalized ground truth value as input also, sigmoid is right.

            return gened_p, gened_theta

    def discriminator_rb(self, p, theta, reuse=False):  # relation-based diecriminator
        with tf.variable_scope("discriminator_rb", reuse=reuse):

            input_tensor = tf.concat(p, theta, axis=1)

            k = self.num_classes + self.num_geopara

            # encoder multi layersm
            d_e_w1 = tf.get_variable('d_e_w1', shape=[k, (k) * 2], initializer=self.weight_init)
            d_e_b1 = tf.get_variable('d_e_b1', shape=[(k) * 2], initializer=self.weight_init)
            d_e_w2 = tf.get_variable('d_e_w2', shape=[(k) * 2, (k) * 2], initializer=self.weight_init)
            d_e_b2 = tf.get_variable('d_e_b2', shape=[(k) * 2], initializer=self.weight_init)
            d_e_w3 = tf.get_variable('d_e_w3', shape=[(k) * 2, k], initializer=self.weight_init)
            d_e_b3 = tf.get_variable('d_e_b3', shape=[k], initializer=self.weight_init)

            # decoder multi layers
            d_d_w1 = tf.get_variable('d_d_w1', shape=[k, k // 2], initializer=self.weight_init)
            d_d_b1 = tf.get_variable('d_d_b1', shape=[k // 2], initializer=self.weight_init)
            d_d_w2 = tf.get_variable('d_d_w2', shape=[k // 2, max(k // 4, 1)], initializer=self.weight_init)
            d_d_b2 = tf.get_variable('d_d_b2', shape=[max(k // 4, 1)], initializer=self.weight_init)
            d_d_w3 = tf.get_variable('d_d_w3', shape=[max(k // 4, 1), 1], initializer=self.weight_init)
            d_d_b3 = tf.get_variable('d_d_b3', shape=[1],
                                     initializer=self.weight_init)  # true, or false? 0.5 threshold by sigmoid

            d_d_p_w = tf.get_variable('d_d_p_w', shape=[self.num_classes, self.num_classes],
                                      initializer=self.weight_init)
            d_d_theta_w = tf.get_variable('d_d_theta_w', shape=[self.num_geopara, self.num_geopara],
                                          initializer=self.weight_init)

            # Embedding FC Layer
            L1 = tf.nn.relu(tf.matmul(input_tensor, d_e_w1) + d_e_b1)
            L2 = tf.nn.relu(tf.matmul(L1, d_e_w2) + d_e_b2)
            relation_layer = dict()
            relation_layer[0] = tf.nn.relu(tf.matmul(L2, d_e_w3) + d_e_b3)  # shape of [self.num_elements, k]

            # since no information about number of relation module, I implemented with just one relation block
            d_sa_hi_w = [tf.get_variable('d_sa_hi_w' + str(i), shape=[k, k], initializer=self.weight_init) for i in
                         range(2)]
            d_sa_hj_w = [tf.get_variable('d_sa_hj_w' + str(i), shape=[k, k], initializer=self.weight_init) for i in
                         range(2)]
            d_sa_u_w = [tf.get_variable('d_sa_u_w' + str(i), shape=[k, k], initializer=self.weight_init) for i in
                        range(2)]
            d_sa_outer_w = [tf.get_variable('d_sa_outer_w' + str(i), shape=[k, k], initializer=self.weight_init) for i
                            in range(2)]

            # Self-Attention + Residual connection Block
            for n in range(1):  # total 2 layer -> forward for 1 steps.
                for i in range(self.num_elements):
                    d_sa_hi = tf.matmul(tf.reshape(relation_layer[n][i], shape=[1, k]), d_sa_hi_w[n])
                    sigma_list = []
                    for j in range(self.num_elements):
                        if i != j:
                            d_sa_hj = tf.matmul(tf.reshape(relation_layer[n][j], shape=[1, k]), d_sa_hj_w[n])
                            H = tf.tensordot(d_sa_hi, d_sa_hj, 1)
                            u = tf.matmul(tf.reshape(relation_layer[n][j], shape=[1, k]), d_sa_u_w[n])
                            sigma_list.append(tf.math.scalar_mul(H, u))

                    sigma = tf.math.add_n(sigma_list)

                    relation_layer[n + 1][i] = tf.reshape(tf.matmul(sigma * float(1. / self.num_elements), d_sa_outer_w[
                        n]))  # no shortcut connection   # have to fix this with tf.scatter_nd

            # Shape of relation_layer[1] is [self.num_elements, k]
            L0_ = tf.math.reduce_max(relation_layer[1], axis=0)  # max pooling used in (Charles et al., 2017)

            # decoder part
            L1_ = tf.nn.relu(tf.matmul(L0_, d_d_w1) + d_d_b1)
            L2_ = tf.nn.relu(tf.matmul(L1_, d_d_w2) + d_d_b2)
            D_logit = tf.nn.relu(tf.matmul(L2_, d_d_w3) + d_d_b3)
            D_prob = tf.nn.sigmoid(D_logit)

            return D_prob, D_logit

    def point_renderer(self, p, theta):
        I = {}

        for x in self.geo_range[0]:
            I[x] = {}
            for y in self.geo_range[1]:
                I[x][y] = {}
                for c in range(self.num_classes):
                    # theta[i][0] is x_i, theta[i][1] is y_i.
                    I[x][y][c] = tf.math.reduce_max(np.array([tf.nn.relu(
                        1. - tf.math.abs(x - theta[i][0])) * tf.nn.relu(1. - tf.math.abs(y - theta[i][1])) * p[i][c] for
                                                              i in range(self.num_elements)]), axis=0)
        # 이렇게 구현하면 에러 남. gradient를 뒤로 flow 시키기 위해선 tensor의 한 element씩 떼와서 연산을 하면 안되고, 행렬곱 등으로 TF가 제공하는 함수를 이용해 연산해야함. 추후 수정 예정. 일단 psuedo code 로만 구현해놓음.

        temp_2 = []
        for k_x, v_x in I.items():
            temp_1 = []
            for k_y, v_y in v_x.items():
                temp_0 = [v_c for k_c, v_c in v_y.items()]
                temp_1.append(temp_0)
            temp_2.append(temp_1)

        return temp_2

    def rectangle_renderer(self, p, theta):
        I = {}

        for x in self.geo_range[0]:
            I[x] = {}
            for y in self.geo_range[1]:
                I[x][y] = {}
                for c in range(self.num_classes):
                    # theta[i] -> [x_L, y_T, x_R, y_B]

                    tf.math.reduce_max([tf.math.reduce_max(np.array([tf.nn.relu(
                        1. - tf.math.abs(x - theta[i][0])) * tf.reduce_min([tf.nn.relu(y - theta[i][1]), 1],
                                                                           axis=0) * tf.reduce_min(
                        [tf.nn.relu(theta[i][3] - y), 1], axis=0),
                                                                     tf.nn.relu(1. - tf.math.abs(
                                                                         x - theta[i][2])) * tf.reduce_min(
                                                                         [tf.nn.relu(y - theta[i][1]), 1],
                                                                         axis=0) * tf.reduce_min(
                                                                         [tf.nn.relu(theta[i][3] - y), 1], axis=0),
                                                                     tf.nn.relu(1. - tf.math.abs(
                                                                         y - theta[i][1])) * tf.reduce_min(
                                                                         [tf.nn.relu(x - theta[i][0]), 1],
                                                                         axis=0) * tf.reduce_min(
                                                                         [tf.nn.relu(theta[i][2] - x), 1], axis=0),
                                                                     tf.nn.relu(1. - tf.math.abs(
                                                                         y - theta[i][3])) * tf.reduce_min(
                                                                         [tf.nn.relu(x - theta[i][0]), 1],
                                                                         axis=0) * tf.reduce_min(
                                                                         [tf.nn.relu(theta[i][2] - x), 1], axis=0)])) *
                                        p[i][c] for i in range(self.num_elements)], axis=0)

        # 이렇게 구현하면 에러 남. gradient를 뒤로 flow 시키기 위해선 tensor의 한 element씩 떼와서 연산을 하면 안되고, 행렬곱 등으로 TF가 제공하는 함수를 이용해 연산해야함. 추후 수정 예정. 일단 psuedo code 로만 구현해놓음.
        # 예를 들어 좌표값만 있는 X 라는 텐서를 만든 후 일괄적으로 빼 주고.. 이런 식으로 구현해야 한다고 생각.

        temp_2 = []
        for k_x, v_x in I.items():
            temp_1 = []
            for k_y, v_y in v_x.items():
                temp_0 = [v_c for k_c, v_c in v_y.items()]
                temp_1.append(temp_0)
            temp_2.append(temp_1)

        return temp_2

    '''
    def triangle_renderer(self, p, theta):
        I = {}

        for x in self.geo_range[0]:
            I[x] = {}
            for y in self.geo_range[1]:
                I[x][y] = {}
                for c in range(self.num_classes):
                    # theta[i] -> [x_L, y_T, x_R, y_B]

                    tf.math.reduce_max( tf.math.reduce_max( np.array([tf.nn.relu(1 - tf.math.abs( y - (theta[i][3] - theta[i][1])))* (x - theta[i][0]) / (theta[i][2] - theta[i][0])) - theta[i][0] ) * tf.reduce_min([tf.nn.relu(x - theta[i][0]), 1], axis=0) * tf.reduce_min([tf.nn.relu(theta[i][2] - x), 1], axis=0),
                                        tf.nn.relu(1 - tf.math.abs(y - (theta[i][3] - theta[i][1]))) * (x - theta[i][0]) / (theta[i][2] - theta[i][0])) - theta[i][0] ) *tf.reduce_min([tf.nn.relu(x - theta[i][0]), 1], axis=0) * tf.reduce_min([tf.nn.relu(theta[i][2] - x), 1], axis=0),                            


                    # 이렇게 구현하면 에러 남. gradient를 뒤로 flow 시키기 위해선 tensor의 한 element씩 떼와서 연산을 하면 안되고, 행렬곱 등으로 TF가 제공하는 함수를 이용해 연산해야함. 추후 수정 예정. 일단 psuedo code 로만 구현해놓음.

        temp_2 = []
        for k_x, v_x in I.items():
            temp_1 = []
            for k_y, v_y in v_x.items():
                temp_0 = [v_c for k_c, v_c in v_y.items()]
                temp_1.append(temp_0)
            temp_2.append(temp_1)

        return temp_2
    '''

    def discriminator_wr(self, p, theta, geo_type='point', reuse=False):  # wireframe rendering discriminator
        with tf.variable_scope("discriminator_wr", reuse=reuse):

            p_variable = tf.get_variable('p_variable', shape=[self.num_elements, self.num_classes])
            theta_variable = tf.get_variable('theta_variable', shape=[self.num_elements, self.num_classes])
            # input_tensor = tf.concat(p, theta, axis=1)
            k = self.num_classes + self.num_geopara

            if geo_type == 'point':
                rendered = self.point_renderer(p_variable, theta_variable)
            if geo_type == 'rectangle':
                rendered = self.rectangle_renderer(p_variable, theta_variable)
            '''
            if geo_type = 'triangle':
                rendered = triangle_renderer()
            '''

            conv1 = tf.layers.conv2d(inputs=rendered, filters=4, kernel_size=3,
                                     padding="SAME", activation=tf.nn.relu)

            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2,
                                            padding="SAME", strides=2)

            conv2 = tf.layers.conv2d(inputs=pool1, filters=8, kernel_size=3,
                                     padding="SAME", activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2,
                                            padding="SAME", strides=2)

            conv3 = tf.layers.conv2d(inputs=pool2, filters=16, kernel_size=3,
                                     padding="SAME", activation=tf.nn.relu)
            pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=2,
                                            padding="SAME", strides=2)

            flat = tf.reshape(pool3, [-1, (math.ceil(math.ceil(math.ceil(len(rendered) / 2.) / 2.) / 2.)) * (
                math.ceil(math.ceil(math.ceil(len(rendered[0]) / 2.) / 2.) / 2.)) * 16])
            dense = tf.layers.dense(inputs=flat, units=1, activation=tf.nn.relu)
            prob = tf.nn.sigmoid(dense)

            return prob, dense

    def build_model(self):

        ### ground-truth value also should be sigmoided.

        self.p_input = tf.placeholder(tf.float32, shape=[None, self.num_classes])
        self.theta_input = tf.placeholder(tf.float32, shape=[None, self.num_geopara])

        # fake data generation
        G_p, G_theta = self.generator(reuse=False)

        if self.which_discriminator == 'rb':
            # real data
            D_real_rb, D_real_logits_rb = self.discriminator_rb(self.p_input, self.theta_input, reuse=False)

            # fake data
            D_fake_rb, D_fake_logits_rb = self.discriminator_rb(G_p, G_theta, reuse=False)

            # loss of discriminator
            D_loss_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits_rb, labels=tf.ones_like(D_real_rb)))
            D_loss_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits_rb, labels=tf.zeros_like(D_fake_rb)))
            self.D_loss = D_loss_real + D_loss_fake

            # loss of generator
            self.G_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits_rb, labels=tf.ones_like(D_fake_rb)))
        elif self.which_discriminator == 'rb':
            # real data
            D_real_wr, D_real_logits_wr = self.discriminator_wr(self.p_input, self.theta_input, reuse=False)

            # fake data
            D_fake_wr, D_fake_logits_wr = self.discriminator_wr(G_p, G_theta, reuse=False)

            # loss of discriminator
            D_loss_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits_wr, labels=tf.ones_like(D_real_wr)))
            D_loss_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits_wr, labels=tf.zeros_like(D_fake_wr)))
            self.D_loss = D_loss_real + D_loss_fake

            # loss of generator
            self.G_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits_wr, labels=tf.ones_like(D_fake_wr)))
        else:
            raise NotImplementedError

        # train을 따로 시켜줘야 하므로 discriminator와 generator의 variable을 나눠줌. tensorflow 구현 문제로 인해 완벽히 나누진 못함
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]

        # optimizer
        if self.which_discriminator == 'rb':
            self.D_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1, beta2=self.beta2).minimize(
                self.D_loss, var_list=d_vars)
        elif self.which_discriminator == 'rb':
            self.D_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1, beta2=self.beta2).minimize(
                self.D_loss, var_list=d_vars)
        else:
            raise NotImplementedError

        self.G_optim = tf.train.AdamOptimizer(self.learning_rate * 5, beta1=self.beta1, beta2=self.beta2) \
            .minimize(self.G_loss, var_list=g_vars)

    def train(self):
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        for epoch in range(self.epoch):
            _, d_loss = self.sess.run([self.D_optim, self.D_loss],
                                      feed_dict={self.p_input: self.data_p, self.theta_input: self.data_theta})

            _, g_loss = self.sess.run([self.G_optim, self.G_loss])


if __name__ == '__main__':
    layoutgan = layoutGAN(epoch=1)
    layoutgan.train(layoutgan)