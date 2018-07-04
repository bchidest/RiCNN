import tensorflow as tf
import math
import numpy as np


def rotated_weights(weights, n_rotations, interpolation='NEAREST'):
    '''Creates new weight variable by rotation and concatenating weights.
       NOTE: This is used for DFT transition layer, not convolutional
         layers. For convolutional layers, rotations are applied in
         rotational_conv2d functions.'''
    assert interpolation in ['BILINEAR', 'NEAREST']
    assert n_rotations in [4, 8]
    rot_angle = 2*np.pi / n_rotations
    weights_rotated = []
    weights = tf.transpose(weights, [3, 0, 1, 2])
    for r in range(1, n_rotations):
        weights_rotated.append(
                tf.contrib.image.rotate(weights, r*rot_angle,
                                        interpolation=interpolation))
    weights = tf.concat([weights] + weights_rotated, axis=0)
    weights = tf.transpose(weights, [1, 2, 3, 0])
    return weights


def calculate_dft_output_size(n_filters, n_rotations):
    return n_filters*n_rotations


def calculate_reconv_midpoints(input_size, weights_size, stride=1,
                               padding='VALID'):
    input_size = int(input_size)
    weights_size = int(weights_size)
    weights_radius = weights_size / 2  # radius will therefore exclude center pixel
    mid = input_size / 2
    if padding == 'SAME':
        stride_remainder = (input_size / 2) % stride
    else:
        stride_remainder = (input_size / 2 - weights_radius - 1) % stride
    if input_size % 2 == 0:
        mid_a = mid
        mid_b = mid + stride_remainder
    else:
        mid_b = mid + 1 + stride_remainder
        mid_a = mid - stride_remainder
    return mid, mid_a, mid_b


def calculate_reconv_output_size(input_size, weights_size, stride=1,
                                 pool_stride=1, padding='VALID'):
    assert padding in ['VALID']
    input_size = int(input_size)
    weights_size = int(weights_size)
    weights_radius = weights_size / 2  # radius will therefore exclude center pixel
    _, mid_a, _ = calculate_reconv_midpoints(input_size, weights_size, stride,
                                             padding)
    output_size = 2*int(math.ceil((mid_a - weights_radius) /
                        float(stride)))

    if input_size % 2 != 0:
        output_size += 1
    if pool_stride == 2:
        output_size = output_size / 2
    return output_size


def rotation_equivariant_conv2d(x, W, x_shape, W_shape, n_rotations, stride=1,
                                padding='VALID', interpolation='NEAREST'):
    # Only VALID padding is currently tested
    assert padding in ['VALID']
    assert interpolation in ['BILINEAR', 'NEAREST']
    assert n_rotations in [4, 8]
    batch_size = x_shape[0]
    W_s = W_shape[0]  # filter will always be odd
    W_radius = W_s / 2  # radius will therefore exclude center pixel
    W_d = W_shape[3]
    x_s = x_shape[1]  # could be odd or even
    stride_remainder = (x_s / 2 - W_radius - 1) % stride
    mid = x_s / 2
    # TODO: incorporate the following line
    if x_s % 2 == 0:
        mid_a = mid
        mid_b = mid + stride_remainder
    else:
        mid_b = mid + 1 + stride_remainder
        mid_a = mid - stride_remainder

    # If rotations of 90 degrees
    if n_rotations == 4:
        W_90 = tf.expand_dims(tf.image.rot90(W[:, :, :, 0]), -1)
        W_180 = tf.expand_dims(tf.image.rot90(W_90[:, :, :, 0]), -1)
        W_270 = tf.expand_dims(tf.image.rot90(W_180[:, :, :, 0]), -1)
        for d in range(1, W_d):
            W_90 = tf.concat([W_90, tf.expand_dims(
                tf.image.rot90(W[:, :, :, d]), -1)],
                             axis=3)
            W_180 = tf.concat([W_180, tf.expand_dims(
                tf.image.rot90(W_90[:, :, :, d]), -1)],
                              axis=3)
            W_270 = tf.concat([W_270, tf.expand_dims(
                tf.image.rot90(W_180[:, :, :, d]), -1)],
                              axis=3)
        # If input size is odd
        if x_s % 2 == 1:
            x_vert1_a = tf.nn.conv2d(
                    x[:, 0:mid_a + W_radius,
                        mid - W_radius: mid + W_radius + 1, :],
                    W, strides=[1, stride, stride, 1], padding='VALID')
            x_vert1_b = tf.nn.conv2d(
                    x[:, 0:mid_a + W_radius,
                        mid - W_radius: mid + W_radius + 1, :],
                    W_270, strides=[1, stride, stride, 1], padding='VALID')
            x_vert2_a = tf.nn.conv2d(
                    x[:, mid_b - W_radius:,
                        mid - W_radius: mid + W_radius + 1, :],
                    W_180, strides=[1, stride, stride, 1], padding='VALID')
            x_vert2_b = tf.nn.conv2d(
                    x[:, mid_b - W_radius:,
                        mid - W_radius: mid + W_radius + 1, :],
                    W_90, strides=[1, stride, stride, 1], padding='VALID')
            x_hor1_a = tf.nn.conv2d(
                    x[:, mid - W_radius: mid + W_radius + 1,
                        0:mid_a + W_radius, :],
                    W, strides=[1, stride, stride, 1], padding='VALID')
            x_hor1_b = tf.nn.conv2d(
                    x[:, mid - W_radius: mid + W_radius + 1,
                        0:mid_a + W_radius, :],
                    W_90, strides=[1, stride, stride, 1], padding='VALID')
            x_hor2_a = tf.nn.conv2d(
                    x[:, mid - W_radius: mid + W_radius + 1,
                        mid_b - W_radius:, :],
                    W_180, strides=[1, stride, stride, 1], padding='VALID')
            x_hor2_b = tf.nn.conv2d(
                    x[:, mid - W_radius: mid + W_radius + 1,
                        mid_b - W_radius:, :],
                    W_270, strides=[1, stride, stride, 1], padding='VALID')
            x_center_a = tf.nn.conv2d(
                    x[:, mid - W_radius: mid + W_radius + 1,
                        mid - W_radius: mid + W_radius + 1, :],
                    W, strides=[1, stride, stride, 1], padding='VALID')
            x_center_b = tf.nn.conv2d(
                    x[:, mid - W_radius: mid + W_radius + 1,
                        mid - W_radius: mid + W_radius + 1, :],
                    W_90, strides=[1, stride, stride, 1], padding='VALID')
            x_center_c = tf.nn.conv2d(
                    x[:, mid - W_radius: mid + W_radius + 1,
                        mid - W_radius: mid + W_radius + 1, :],
                    W_180, strides=[1, stride, stride, 1], padding='VALID')
            x_center_d = tf.nn.conv2d(
                    x[:, mid - W_radius: mid + W_radius + 1,
                        mid - W_radius: mid + W_radius + 1, :],
                    W_270, strides=[1, stride, stride, 1], padding='VALID')

            x_vert1 = tf.concat([tf.expand_dims(x_vert1_a, axis=-1),
                                 tf.expand_dims(x_vert1_b, axis=-1)], axis=4)
            x_vert2 = tf.concat([tf.expand_dims(x_vert2_a, axis=-1),
                                 tf.expand_dims(x_vert2_b, axis=-1)], axis=4)
            x_hor1 = tf.concat([tf.expand_dims(x_hor1_a, axis=-1),
                                tf.expand_dims(x_hor1_b, axis=-1)], axis=4)
            x_hor2 = tf.concat([tf.expand_dims(x_hor2_a, axis=-1),
                                tf.expand_dims(x_hor2_b, axis=-1)], axis=4)
            x_center = tf.concat([tf.expand_dims(x_center_a, axis=-1),
                                  tf.expand_dims(x_center_b, axis=-1),
                                  tf.expand_dims(x_center_c, axis=-1),
                                  tf.expand_dims(x_center_d, axis=-1)], axis=4)
            x_vert1 = tf.reduce_max(x_vert1, reduction_indices=[4])
            x_vert2 = tf.reduce_max(x_vert2, reduction_indices=[4])
            x_hor1 = tf.reduce_max(x_hor1, reduction_indices=[4])
            x_hor2 = tf.reduce_max(x_hor2, reduction_indices=[4])
            x_center = tf.reduce_max(x_center, reduction_indices=[4])

            x_out = tf.nn.conv2d(
                    x[:, 0:mid_a + W_radius, 0:mid_a + W_radius, :],
                    W, strides=[1, stride, stride, 1], padding='VALID')
            x_out = tf.concat([x_out, x_vert1, tf.nn.conv2d(
                x[:, 0:mid_a + W_radius, mid_b - W_radius:, :],
                W_270, strides=[1, stride, stride, 1], padding='VALID')],
                axis=2)
            x_out_temp = tf.nn.conv2d(
                    x[:, mid_b - W_radius:, 0:mid_a + W_radius, :],
                    W_90, strides=[1, stride, stride, 1], padding='VALID')
            x_out_temp = tf.concat([x_out_temp, x_vert2, tf.nn.conv2d(
                x[:, mid_b - W_radius:, mid_b - W_radius:, :],
                W_180, strides=[1, stride, stride, 1], padding='VALID')],
                axis=2)
            x_mid = tf.concat([x_hor1, x_center, x_hor2], axis=2)
            x_out = tf.concat([x_out, x_mid, x_out_temp], axis=1)
        # If input size is even
        else:
            x_out = tf.nn.conv2d(
                    x[:, 0:mid_a + W_radius, 0:mid_a + W_radius, :],
                    W, strides=[1, stride, stride, 1], padding='VALID')
            x_out = tf.concat([x_out, tf.nn.conv2d(
                x[:, 0:mid_a + W_radius, mid_b - W_radius:, :],
                W_270, strides=[1, stride, stride, 1], padding='VALID')],
                axis=2)
            x_out_temp = tf.nn.conv2d(
                    x[:, mid_b - W_radius:, 0:mid_a + W_radius, :],
                    W_90, strides=[1, stride, stride, 1], padding='VALID')
            x_out_temp = tf.concat([x_out_temp, tf.nn.conv2d(
                x[:, mid_b - W_radius:, mid_b - W_radius:, :],
                W_180, strides=[1, stride, stride, 1], padding='VALID')],
                axis=2)
            x_out = tf.concat([x_out, x_out_temp], axis=1)
    # If rotations of 90 degrees
    elif n_rotations == 8:
        W = tf.transpose(W, [3, 0, 1, 2])
        W_45 = tf.contrib.image.rotate(W, np.pi / 4,
                                       interpolation=interpolation)
        W_45 = tf.transpose(W_45, [1, 2, 3, 0])
        W_90 = tf.contrib.image.rotate(W, np.pi / 2,
                                       interpolation=interpolation)
        W_90 = tf.transpose(W_90, [1, 2, 3, 0])
        W_135 = tf.contrib.image.rotate(W, 3*np.pi / 4,
                                        interpolation=interpolation)
        W_135 = tf.transpose(W_135, [1, 2, 3, 0])
        W_180 = tf.contrib.image.rotate(W, np.pi,
                                        interpolation=interpolation)
        W_180 = tf.transpose(W_180, [1, 2, 3, 0])
        W_225 = tf.contrib.image.rotate(W, 5*np.pi / 4,
                                        interpolation=interpolation)
        W_225 = tf.transpose(W_225, [1, 2, 3, 0])
        W_270 = tf.contrib.image.rotate(W, 3*np.pi / 2,
                                        interpolation=interpolation)
        W_270 = tf.transpose(W_270, [1, 2, 3, 0])
        W_315 = tf.contrib.image.rotate(W, 7*np.pi / 4,
                                        interpolation=interpolation)
        W_315 = tf.transpose(W_315, [1, 2, 3, 0])
        W = tf.transpose(W, [1, 2, 3, 0])

        # Create triangular masks in Numpy for masking output of convolutions
        #    of filters with appropriate quadrants. For 8 rotations, each
        #    quadrant is partitioned into an upper and lower triangle and
        #    convolved with two filters, offset by a 45 degree rotation, for
        #    each triangle.
        x_radius_after_conv = int(math.ceil(float((x_s / 2 - W_radius) /
                                            float(stride))))
        mask_np = np.zeros((batch_size, x_radius_after_conv,
                            x_radius_after_conv, W_d), dtype='bool')
        mask_np_90 = np.zeros((batch_size, x_radius_after_conv,
                               x_radius_after_conv, W_d), dtype='bool')
        mask_diag_np = np.zeros((batch_size, x_radius_after_conv,
                                 x_radius_after_conv, W_d), dtype='bool')
        mask_diag_2_np = np.zeros((batch_size, x_radius_after_conv,
                                   x_radius_after_conv, W_d), dtype='bool')

        for b in range(batch_size):
            for i in range(x_radius_after_conv):
                for j in range(x_radius_after_conv):
                    if j > i:
                        mask_np[b, i, j, :] = 1
                    elif j == i:
                        mask_diag_np[b, i, j, :] = 1
                    if j < x_radius_after_conv - i - 1:
                        mask_np_90[b, i, j, :] = 1
                    elif j == x_radius_after_conv - i - 1:
                        mask_diag_2_np[b, i, j, :] = 1

        # Convert Numpy masks to tensors.
        # TODO: Should the generation of these masks be done external to this
        #   function and then passed in? Is this currently doing unnecessary
        #   recomputation? I don't think so, but not certain.
        mask_0 = tf.convert_to_tensor(mask_np)
        mask_90 = tf.convert_to_tensor(mask_np_90)
        mask_225 = mask_0
        mask_315 = mask_90
        mask_diag_ullr = tf.convert_to_tensor(mask_diag_np)
        mask_diag_llur = tf.convert_to_tensor(mask_diag_2_np)

        # Split input feature map into quadrants (ul, ur, ll, lr).
        x_ul = x[:, 0:mid_a + W_radius, 0:mid_a + W_radius, :]
        x_ur = x[:, 0:mid_a + W_radius, mid_b - W_radius:, :]
        x_lr = x[:, mid_b - W_radius:, mid_b - W_radius:, :]
        x_ll = x[:, mid_b - W_radius:, 0:mid_a + W_radius, :]

        # Convolve rotated filters with appopriate quadrant.
        # Ex: _ur_270 is the output of the ur quadrant convoved with the
        #     set of filters rotated by 270 degrees
        x_ul_0 = tf.nn.conv2d(x_ul, W, strides=[1, stride, stride, 1],
                              padding='VALID')
        x_ul_45 = tf.nn.conv2d(x_ul, W_45, strides=[1, stride, stride, 1],
                               padding='VALID')
        x_ur_270 = tf.nn.conv2d(x_ur, W_270, strides=[1, stride, stride, 1],
                                padding='VALID')
        x_ur_315 = tf.nn.conv2d(x_ur, W_315, strides=[1, stride, stride, 1],
                                padding='VALID')
        x_lr_180 = tf.nn.conv2d(x_lr, W_180, strides=[1, stride, stride, 1],
                                padding='VALID')
        x_lr_225 = tf.nn.conv2d(x_lr, W_225, strides=[1, stride, stride, 1],
                                padding='VALID')
        x_ll_90 = tf.nn.conv2d(x_ll, W_90, strides=[1, stride, stride, 1],
                               padding='VALID')
        x_ll_135 = tf.nn.conv2d(x_ll, W_135, strides=[1, stride, stride, 1],
                                padding='VALID')

        # Mask convolution of each quadrant according to rotation region
        x_ul = tf.where(tf.logical_or(mask_0, tf.logical_and(mask_diag_ullr,
                                      tf.greater(x_ul_0, x_ul_45))),
                        x_ul_0, x_ul_45)
        x_ll = tf.where(tf.logical_or(mask_90, tf.logical_and(mask_diag_llur,
                                      tf.greater(x_ll_90, x_ll_135))),
                        x_ll_90, x_ll_135)
        x_lr = tf.where(tf.logical_or(mask_225, tf.logical_and(mask_diag_ullr,
                                      tf.greater(x_lr_225, x_lr_180))),
                        x_lr_225, x_lr_180)
        x_ur = tf.where(tf.logical_or(mask_315, tf.logical_and(mask_diag_llur,
                                      tf.greater(x_ur_315, x_ur_270))),
                        x_ur_315, x_ur_270)

        # If input size is odd
        if x_s % 2 == 1:
            x_vert1_a = tf.nn.conv2d(
                    x[:, 0:mid_a + W_radius,
                        mid - W_radius: mid + W_radius + 1, :],
                    W, strides=[1, stride, stride, 1], padding='VALID')
            x_vert1_b = tf.nn.conv2d(
                    x[:, 0:mid_a + W_radius,
                        mid - W_radius: mid + W_radius + 1, :],
                    W_315, strides=[1, stride, stride, 1], padding='VALID')
            x_vert2_a = tf.nn.conv2d(
                    x[:, mid_b - W_radius:,
                        mid - W_radius: mid + W_radius + 1, :],
                    W_180, strides=[1, stride, stride, 1], padding='VALID')
            x_vert2_b = tf.nn.conv2d(
                    x[:, mid_b - W_radius:,
                        mid - W_radius: mid + W_radius + 1, :],
                    W_135, strides=[1, stride, stride, 1], padding='VALID')
            x_hor1_a = tf.nn.conv2d(
                    x[:, mid - W_radius: mid + W_radius + 1,
                        0:mid_a + W_radius, :],
                    W_45, strides=[1, stride, stride, 1], padding='VALID')
            x_hor1_b = tf.nn.conv2d(
                    x[:, mid - W_radius: mid + W_radius + 1,
                        0:mid_a + W_radius, :],
                    W_90, strides=[1, stride, stride, 1], padding='VALID')
            x_hor2_a = tf.nn.conv2d(
                    x[:, mid - W_radius: mid + W_radius + 1,
                        mid_b - W_radius:, :],
                    W_225, strides=[1, stride, stride, 1], padding='VALID')
            x_hor2_b = tf.nn.conv2d(
                    x[:, mid - W_radius: mid + W_radius + 1,
                        mid_b - W_radius:, :],
                    W_270, strides=[1, stride, stride, 1], padding='VALID')
            x_center_a = tf.nn.conv2d(
                    x[:, mid - W_radius: mid + W_radius + 1,
                        mid - W_radius: mid + W_radius + 1, :],
                    W, strides=[1, stride, stride, 1], padding='VALID')
            x_center_b = tf.nn.conv2d(
                    x[:, mid - W_radius: mid + W_radius + 1,
                        mid - W_radius: mid + W_radius + 1, :],
                    W_45, strides=[1, stride, stride, 1], padding='VALID')
            x_center_c = tf.nn.conv2d(
                    x[:, mid - W_radius: mid + W_radius + 1,
                        mid - W_radius: mid + W_radius + 1, :],
                    W_90, strides=[1, stride, stride, 1], padding='VALID')
            x_center_d = tf.nn.conv2d(
                    x[:, mid - W_radius: mid + W_radius + 1,
                        mid - W_radius: mid + W_radius + 1, :],
                    W_135, strides=[1, stride, stride, 1], padding='VALID')
            x_center_e = tf.nn.conv2d(
                    x[:, mid - W_radius: mid + W_radius + 1,
                        mid - W_radius: mid + W_radius + 1, :],
                    W_180, strides=[1, stride, stride, 1], padding='VALID')
            x_center_f = tf.nn.conv2d(
                    x[:, mid - W_radius: mid + W_radius + 1,
                        mid - W_radius: mid + W_radius + 1, :],
                    W_225, strides=[1, stride, stride, 1], padding='VALID')
            x_center_g = tf.nn.conv2d(
                    x[:, mid - W_radius: mid + W_radius + 1,
                        mid - W_radius: mid + W_radius + 1, :],
                    W_270, strides=[1, stride, stride, 1], padding='VALID')
            x_center_h = tf.nn.conv2d(
                    x[:, mid - W_radius: mid + W_radius + 1,
                        mid - W_radius: mid + W_radius + 1, :],
                    W_315, strides=[1, stride, stride, 1], padding='VALID')

            x_vert1 = tf.concat([tf.expand_dims(x_vert1_a, axis=-1),
                                 tf.expand_dims(x_vert1_b, axis=-1)], axis=4)
            x_vert2 = tf.concat([tf.expand_dims(x_vert2_a, axis=-1),
                                 tf.expand_dims(x_vert2_b, axis=-1)], axis=4)
            x_hor1 = tf.concat([tf.expand_dims(x_hor1_a, axis=-1),
                                tf.expand_dims(x_hor1_b, axis=-1)], axis=4)
            x_hor2 = tf.concat([tf.expand_dims(x_hor2_a, axis=-1),
                                tf.expand_dims(x_hor2_b, axis=-1)], axis=4)
            x_center = tf.concat([tf.expand_dims(x_center_a, axis=-1),
                                  tf.expand_dims(x_center_b, axis=-1),
                                  tf.expand_dims(x_center_c, axis=-1),
                                  tf.expand_dims(x_center_d, axis=-1),
                                  tf.expand_dims(x_center_e, axis=-1),
                                  tf.expand_dims(x_center_f, axis=-1),
                                  tf.expand_dims(x_center_g, axis=-1),
                                  tf.expand_dims(x_center_h, axis=-1)], axis=4)
            x_vert1 = tf.reduce_max(x_vert1, reduction_indices=[4])
            x_vert2 = tf.reduce_max(x_vert2, reduction_indices=[4])
            x_hor1 = tf.reduce_max(x_hor1, reduction_indices=[4])
            x_hor2 = tf.reduce_max(x_hor2, reduction_indices=[4])
            x_center = tf.reduce_max(x_center, reduction_indices=[4])

            # Merge quadrants convolutions into a single feature map
            x_out_u = tf.concat([x_ul, x_vert1, x_ur], axis=2)
            x_out_l = tf.concat([x_ll, x_vert2, x_lr], axis=2)
            x_mid = tf.concat([x_hor1, x_center, x_hor2], axis=2)
            x_out = tf.concat([x_out_u, x_mid, x_out_l], axis=1)
        # If input size is even
        else:
            # Merge quadrants convolutions into a single feature map
            x_out_u = tf.concat([x_ul, x_ur], axis=2)
            x_out_l = tf.concat([x_ll, x_lr], axis=2)
            x_out = tf.concat([x_out_u, x_out_l], axis=1)
    else:
        raise ValueError('Rotation equivariant convolution is only ' +
                         'implemented for rotations of 45 or 90 degrees!')
    return x_out


def dft2d_transition(layer, n_rotations, batch_size, conv_size,
                     n_filters_previous_layer, n_nodes):
    # Reshape output to be [batch_size, n_nodes, n_rotations]
    layer = tf.transpose(layer, [1, 0])
    layer = tf.reshape(layer, [n_rotations,
                               n_nodes, batch_size])
    layer = tf.transpose(layer, [2, 1, 0])
    # DFT to enforce rotational invariance
    layer = tf.cast(layer, tf.complex64)
    layer_fft = tf.abs(tf.fft2d(layer))
    # Normalize DFT output
    layer_fft = layer_fft / (conv_size*conv_size*n_filters_previous_layer*n_rotations)
    output_size = calculate_dft_output_size(n_nodes, n_rotations)
    layer_fft = tf.reshape(layer_fft, [batch_size, output_size])
    return layer_fft


def conv_to_circular_shift(layer, weights, biases, n_rotations, padding='VALID'):
    weights = rotated_weights(weights, n_rotations)
    biases = tf.tile(biases, [n_rotations])
    layer = tf.nn.conv2d(layer, weights, strides=[1, 1, 1, 1], padding=padding)\
        + biases
    # Output should be [batch_size, 1, 1, n_nodes*n_rotations], so squeeze
    layer = tf.squeeze(layer)
    return layer
