import tensorflow as tf
import numpy as np
import ricnn

x_size = 15
x_depth = 3
batch_size = 10

with tf.Graph().as_default():
    x = tf.placeholder(tf.float32, [batch_size, x_size, x_size, x_depth])

    # Rotation-equivariant convolution w/ stride
    filter_size = 3
    n_filters = 16
    n_rotations = 4
    stride = 2
    pool_stride = 1
    weights = tf.get_variable('weights-conv1', [filter_size, filter_size,
                                                x_depth, n_filters])
    biases = tf.get_variable('biases-conv1', [n_filters])
    output = ricnn.rotation_equivariant_conv2d(
            x, weights,
            [batch_size, x_size, x_size, x_depth],
            [filter_size, filter_size, x_depth, n_filters],
            n_rotations, stride=stride) + biases
    output = tf.nn.relu(output)
    output_size = ricnn.calculate_reconv_output_size(
            x_size, filter_size, stride=stride)
    n_filters_previous_layer = n_filters

    # Rotation-equivariant convolution w/ max pooling
    filter_size = 3
    n_filters = 16
    n_rotations = 4
    stride = 1
    pool_stride = 2
    weights = tf.get_variable('weights-conv2', [filter_size, filter_size,
                                                n_filters_previous_layer, n_filters])
    biases = tf.get_variable('biases-conv2', [n_filters])
    output = ricnn.rotation_equivariant_conv2d(
            output, weights,
            [batch_size, output_size, output_size, n_filters_previous_layer],
            [filter_size, filter_size, n_filters_previous_layer, n_filters],
            n_rotations, stride=stride) + biases
    output = tf.nn.max_pool(output, [1, pool_stride, pool_stride, 1],
                            [1, pool_stride, pool_stride, 1], padding='VALID')
    output = tf.nn.relu(output)
    output_size = ricnn.calculate_reconv_output_size(
            output_size, filter_size, pool_stride=pool_stride)
    n_filters_previous_layer = n_filters

    # Rotation-invariant Conv-to-Full transition with the 2D-DFT
    n_filters = 16
    n_rotations = 4
    filter_size = output_size
    weights = tf.get_variable('weights-dft', [filter_size, filter_size,
                                              n_filters_previous_layer, n_filters])
    biases = tf.get_variable('biases-dft', [n_filters])
    output = ricnn.conv_to_circular_shift(output, weights, biases, n_rotations)
    output = tf.nn.relu(output)
    output = ricnn.dft2d_transition(output, n_rotations, batch_size, output_size,
                                    n_filters_previous_layer, n_filters)
    dft_size = ricnn.calculate_dft_output_size(n_filters, n_rotations)

    # Fully-connected layer
    n_nodes = 5
    weights = tf.get_variable('weights-full', [dft_size, n_nodes])
    biases = tf.get_variable('biases-full', [n_nodes])
    y = tf.matmul(output, weights) + biases

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    x_random = np.random.rand(batch_size, x_size, x_size, x_depth)
    y_eval = sess.run(y, feed_dict={x: x_random})
    sess.close()

print(y_eval)
