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
    # inputs = inputs.numpy()
    for idx, i in enumerate(inputs):
        self_attention = np.zeros(i.size())
        for jdx, j in enumerate(inputs):
            if idx == jdx:
                continue
            u = layers.Dense(
                self.feature_size * 2 * 2, input_shape=self.feature_size * 2 * 2, activation='relu')(j)
            i_reshape = np.reshape(i.size(), 1)
            j_reshape = np.reshape(j.size(), 1)
            dot = layers.dot([(i_reshape * psi).t(), j_reshape * phi], axes=1)
            self_attention += dot * u
        f_prime.append(w_r * (self_attention / self.num_elements) + i)
    return f_prime


def load_mnist(thresh=200):
    """This function loads the MNIST dataset from TensorFlow official Datasets."""
    # Download from Google
    mnist_builder = tfds.builder('mnist')
    mnist_builder.download_and_prepare()
    # Transfer to TensorFlow `Dataset` object.
    datasets = mnist_builder.as_dataset()
    train_dataset = datasets['train']
    # print(isinstance(train_dataset, tf.data.Dataset))
    # Transfer MNIST images to tensors with geometry information in the iterator.
    iterator = tf.compat.v1.data.make_one_shot_iterator(train_dataset)
    # TODO: Find the API in r2.0 to feed iterable data to the model.
    # Get one element from the iterator.
    element = iterator.get_next()['image']
    # Pick the pixels (as a 2D numpy array).
    pixels = element.numpy()[:, :, 0]
    # For each image, the layout vector shape is exactly same as the image.
    # 0 -> black, 1 -> white.
    layout_vector = np.zeros((28, 28))
    for i in range(0, 28):
        for j in range(0, 28):
            if pixels[i, j] >= thresh:
                layout_vector[i, j] = 1
            else:
                layout_vector[i, j] = 0
    print(layout_vector)
