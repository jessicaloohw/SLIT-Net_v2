# SLIT-Net v2
# DOI: 10.1167/tvst.10.12.2

import tensorflow as tf
import numpy as np


def segment(x1, mode, NUM_FEATURES=16, POOL_SIZE=(4,4,4,4), REGULARIZATION_WEIGHT=0.0):
    """
    x1= [BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, 1]
    mode = True (when training), False (when testing)
    """

    # Other settings:
    initializer = tf.contrib.layers.variance_scaling_initializer()
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_WEIGHT)

    print(x1)

    with tf.variable_scope('encoder1'):

        x2 = tf.layers.conv2d(x1, NUM_FEATURES, [3, 3], [1, 1], "SAME",
                              activation=None,
                              use_bias=True,
                              kernel_initializer=initializer,
                              bias_initializer=tf.zeros_initializer(),
                              kernel_regularizer=regularizer,
                              bias_regularizer=regularizer,
                              name='conv1')
        x2 = tf.layers.conv2d(x2, NUM_FEATURES, [3, 3], [1, 1], "SAME",
                              activation=None,
                              use_bias=True,
                              kernel_initializer=initializer,
                              bias_initializer=tf.zeros_initializer(),
                              kernel_regularizer=regularizer,
                              bias_regularizer=regularizer,
                              name='conv2')
        x2 = tf.layers.batch_normalization(x2, training=mode, name='bn')
        x2 = tf.nn.relu(x2, name='relu')

        print(x2)

    with tf.variable_scope('encoder2'):

        x3 = tf.layers.max_pooling2d(x2, [POOL_SIZE[0], POOL_SIZE[0]], [POOL_SIZE[0], POOL_SIZE[0]], "VALID", name='pool')
        x3 = tf.layers.conv2d(x3, NUM_FEATURES*2, [3, 3], [1, 1], "SAME",
                              activation=None,
                              use_bias=True,
                              kernel_initializer=initializer,
                              bias_initializer=tf.zeros_initializer(),
                              kernel_regularizer=regularizer,
                              bias_regularizer=regularizer,
                              name='conv1')
        x3 = tf.layers.conv2d(x3, NUM_FEATURES*2, [3, 3], [1, 1], "SAME",
                              activation=None,
                              use_bias=True,
                              kernel_initializer=initializer,
                              bias_initializer=tf.zeros_initializer(),
                              kernel_regularizer=regularizer,
                              bias_regularizer=regularizer,
                              name='conv2')
        x3 = tf.layers.batch_normalization(x3, training=mode, name='bn')
        x3 = tf.nn.relu(x3, name='relu')

        print(x3)

    with tf.variable_scope('encoder3'):

        x4 = tf.layers.max_pooling2d(x3, [POOL_SIZE[1], POOL_SIZE[1]], [POOL_SIZE[1], POOL_SIZE[1]], "VALID", name='pool')
        x4 = tf.layers.conv2d(x4, NUM_FEATURES*4, [3, 3], [1, 1], "SAME",
                               activation=None,
                               use_bias=True,
                               kernel_initializer=initializer,
                               bias_initializer=tf.zeros_initializer(),
                               kernel_regularizer=regularizer,
                               bias_regularizer=regularizer,
                               name='conv1')

        x4 = tf.layers.conv2d(x4, NUM_FEATURES*4, [3, 3], [1, 1], "SAME",
                              activation=None,
                              use_bias=True,
                              kernel_initializer=initializer,
                              bias_initializer=tf.zeros_initializer(),
                              kernel_regularizer=regularizer,
                              bias_regularizer=regularizer,
                              name='conv2')
        x4 = tf.layers.batch_normalization(x4, training=mode, name='bn')
        x4 = tf.nn.relu(x4, name='relu')

        print(x4)

    with tf.variable_scope('encoder4'):

        x5 = tf.layers.max_pooling2d(x4, [POOL_SIZE[2], POOL_SIZE[2]], [POOL_SIZE[2], POOL_SIZE[2]], "VALID", name='pool')
        x5 = tf.layers.conv2d(x5, NUM_FEATURES*8, [3, 3], [1, 1], "SAME",
                              activation=None,
                              use_bias=True,
                              kernel_initializer=initializer,
                              bias_initializer=tf.zeros_initializer(),
                              kernel_regularizer=regularizer,
                              bias_regularizer=regularizer,
                              name='conv1')
        x5 = tf.layers.conv2d(x5, NUM_FEATURES*8, [3, 3], [1, 1], "SAME",
                              activation=None,
                              use_bias=True,
                              kernel_initializer=initializer,
                              bias_initializer=tf.zeros_initializer(),
                              kernel_regularizer=regularizer,
                              bias_regularizer=regularizer,
                              name='conv2')
        x5 = tf.layers.batch_normalization(x5, training=mode, name='bn')
        x5 = tf.nn.relu(x5, name='relu')

        print(x5)

    with tf.variable_scope('bottleneck'):

        x6 = tf.layers.max_pooling2d(x5, [POOL_SIZE[3], POOL_SIZE[3]], [POOL_SIZE[3], POOL_SIZE[3]], "VALID", name='pool')
        x6 = tf.layers.conv2d(x6, NUM_FEATURES*16, [3, 3], [1, 1], "SAME",
                               activation=None,
                               use_bias=True,
                               kernel_initializer=initializer,
                               bias_initializer=tf.zeros_initializer(),
                               kernel_regularizer=regularizer,
                               bias_regularizer=regularizer,
                               name='conv1')
        x6 = tf.layers.conv2d(x6, NUM_FEATURES*16, [3, 3], [1, 1], "SAME",
                              activation=None,
                              use_bias=True,
                              kernel_initializer=initializer,
                              bias_initializer=tf.zeros_initializer(),
                              kernel_regularizer=regularizer,
                              bias_regularizer=regularizer,
                              name='conv2')
        x6 = tf.layers.batch_normalization(x6, training=mode, name='bn')
        x6 = tf.nn.relu(x6, name='relu')

        print(x6)

    with tf.variable_scope('skip4'):

        # Upsampling size needs to match bottleneck:
        x6up = tf.layers.conv2d_transpose(x6, NUM_FEATURES*8, [POOL_SIZE[3], POOL_SIZE[3]], [POOL_SIZE[3], POOL_SIZE[3]], "SAME",
                                         activation=None,
                                         use_bias=True,
                                         kernel_initializer=initializer,
                                         bias_initializer=tf.zeros_initializer(),
                                         kernel_regularizer=regularizer,
                                         bias_regularizer=regularizer,
                                         name='deconv')

        print(x6up)

        # Attention (x5, x6):
        a5 = tf.layers.conv2d(x5, NUM_FEATURES*8, [1, 1], [1, 1], "SAME",
                              activation=None,
                              use_bias=True,
                              kernel_initializer=initializer,
                              bias_initializer=tf.zeros_initializer(),
                              kernel_regularizer=regularizer,
                              bias_regularizer=regularizer,
                              name='attn1')
        a5 = tf.layers.batch_normalization(a5, training=mode, name='bn1')

        a6 = tf.layers.conv2d(x6up, NUM_FEATURES*8, [1, 1], [1, 1], "SAME",
                              activation=None,
                              use_bias=True,
                              kernel_initializer=initializer,
                              bias_initializer=tf.zeros_initializer(),
                              kernel_regularizer=regularizer,
                              bias_regularizer=regularizer,
                              name='attn2')
        a6 = tf.layers.batch_normalization(a6, training=mode, name='bn2')

        a56 = tf.nn.relu(a5 + a6, name='relu')
        a56 = tf.layers.conv2d(a56, 1, [1, 1], [1, 1], "SAME",
                               activation=None,
                               use_bias=True,
                               kernel_initializer=initializer,
                               bias_initializer=tf.zeros_initializer(),
                               kernel_regularizer=regularizer,
                               bias_regularizer=regularizer,
                               name='attn3')
        a56 = tf.layers.batch_normalization(a56, training=mode, name='bn3')
        a56 = tf.nn.sigmoid(a56, name='sigmoid')
        a56 = tf.multiply(a56, x5, name="multiply")

        print(a56)

        x7 = tf.concat([a56, x6up], axis=3, name='concat') # encoder4 + skip4

    with tf.variable_scope('decoder4'):

        x7 = tf.layers.conv2d(x7, NUM_FEATURES*8, [3, 3], [1, 1], "SAME",
                              activation=None,
                              use_bias=True,
                              kernel_initializer=initializer,
                              bias_initializer=tf.zeros_initializer(),
                              kernel_regularizer=regularizer,
                              bias_regularizer=regularizer,
                              name='conv1')
        x7 = tf.layers.conv2d(x7, NUM_FEATURES*8, [3, 3], [1, 1], "SAME",
                              activation=None,
                              use_bias=True,
                              kernel_initializer=initializer,
                              bias_initializer=tf.zeros_initializer(),
                              kernel_regularizer=regularizer,
                              bias_regularizer=regularizer,
                              name='conv2')
        x7 = tf.layers.batch_normalization(x7, training=mode, name='bn')
        x7 = tf.nn.relu(x7, name='relu')

        print(x7)

    with tf.variable_scope('skip3'):

        # Upsampling size needs to match encoder 4:
        x7up = tf.layers.conv2d_transpose(x7, NUM_FEATURES*4, [POOL_SIZE[2], POOL_SIZE[2]], [POOL_SIZE[2], POOL_SIZE[2]], "SAME",
                                        activation=None,
                                        use_bias=True,
                                        kernel_initializer=initializer,
                                        bias_initializer=tf.zeros_initializer(),
                                        kernel_regularizer=regularizer,
                                        bias_regularizer=regularizer,
                                        name='deconv')

        print(x7up)

        # Attention (x4, x7):
        a4 = tf.layers.conv2d(x4, NUM_FEATURES*4, [1, 1], [1, 1], "SAME",
                              activation=None,
                              use_bias=True,
                              kernel_initializer=initializer,
                              bias_initializer=tf.zeros_initializer(),
                              kernel_regularizer=regularizer,
                              bias_regularizer=regularizer,
                              name='attn1')
        a4 = tf.layers.batch_normalization(a4, training=mode, name='bn1')

        a7 = tf.layers.conv2d(x7up, NUM_FEATURES*4, [1, 1], [1, 1], "SAME",
                              activation=None,
                              use_bias=True,
                              kernel_initializer=initializer,
                              bias_initializer=tf.zeros_initializer(),
                              kernel_regularizer=regularizer,
                              bias_regularizer=regularizer,
                              name='attn2')
        a7 = tf.layers.batch_normalization(a7, training=mode, name='bn2')

        a47 = tf.nn.relu(a4 + a7, name='relu')
        a47 = tf.layers.conv2d(a47, 1, [1, 1], [1, 1], "SAME",
                               activation=None,
                               use_bias=True,
                               kernel_initializer=initializer,
                               bias_initializer=tf.zeros_initializer(),
                               kernel_regularizer=regularizer,
                               bias_regularizer=regularizer,
                               name='attn3')
        a47 = tf.layers.batch_normalization(a47, training=mode, name='bn3')
        a47 = tf.nn.sigmoid(a47, name='sigmoid')
        a47 = tf.multiply(a47, x4, name="multiply")

        print(a47)

        x8 = tf.concat([a47, x7up], axis=3, name='concat') # encoder3 + skip3

    with tf.variable_scope('decoder3'):

        x8 = tf.layers.conv2d(x8, NUM_FEATURES*4, [3, 3], [1, 1], "SAME",
                              activation=None,
                              use_bias=True,
                              kernel_initializer=initializer,
                              bias_initializer=tf.zeros_initializer(),
                              kernel_regularizer=regularizer,
                              bias_regularizer=regularizer,
                              name='conv1')
        x8 = tf.layers.conv2d(x8, NUM_FEATURES*4, [3, 3], [1, 1], "SAME",
                              activation=None,
                              use_bias=True,
                              kernel_initializer=initializer,
                              bias_initializer=tf.zeros_initializer(),
                              kernel_regularizer=regularizer,
                              bias_regularizer=regularizer,
                              name='conv2')
        x8 = tf.layers.batch_normalization(x8, training=mode, name='bn')
        x8 = tf.nn.relu(x8, name='relu')

        print(x8)

    with tf.variable_scope('skip2'):

        # Upsampling size needs to match encoder3:
        x8up = tf.layers.conv2d_transpose(x8, NUM_FEATURES*2, [POOL_SIZE[1], POOL_SIZE[1]], [POOL_SIZE[1], POOL_SIZE[1]], "SAME",
                                        activation=None,
                                        use_bias=True,
                                        kernel_initializer=initializer,
                                        bias_initializer=tf.zeros_initializer(),
                                        kernel_regularizer=regularizer,
                                        bias_regularizer=regularizer,
                                        name='deconv')

        print(x8up)

        # Attention (x3, x8):
        a3 = tf.layers.conv2d(x3, NUM_FEATURES*2, [1, 1], [1, 1], "SAME",
                              activation=None,
                              use_bias=True,
                              kernel_initializer=initializer,
                              bias_initializer=tf.zeros_initializer(),
                              kernel_regularizer=regularizer,
                              bias_regularizer=regularizer,
                              name='attn1')
        a3 = tf.layers.batch_normalization(a3, training=mode, name='bn1')

        a8 = tf.layers.conv2d(x8up, NUM_FEATURES*2, [1, 1], [1, 1], "SAME",
                              activation=None,
                              use_bias=True,
                              kernel_initializer=initializer,
                              bias_initializer=tf.zeros_initializer(),
                              kernel_regularizer=regularizer,
                              bias_regularizer=regularizer,
                              name='attn2')
        a8 = tf.layers.batch_normalization(a8, training=mode, name='bn2')

        a38 = tf.nn.relu(a3 + a8, name='relu')
        a38 = tf.layers.conv2d(a38, 1, [1, 1], [1, 1], "SAME",
                               activation=None,
                               use_bias=True,
                               kernel_initializer=initializer,
                               bias_initializer=tf.zeros_initializer(),
                               kernel_regularizer=regularizer,
                               bias_regularizer=regularizer,
                               name='attn3')
        a38 = tf.layers.batch_normalization(a38, training=mode, name='bn3')
        a38 = tf.nn.sigmoid(a38, name='sigmoid')
        a38 = tf.multiply(a38, x3, name="multiply")

        print(a38)

        x9 = tf.concat([a38, x8up], axis=3, name='concat') # encoder2 + skip2

    with tf.variable_scope('decoder2'):

        x9 = tf.layers.conv2d(x9, NUM_FEATURES*2, [3, 3], [1, 1], "SAME",
                              activation=None,
                              use_bias=True,
                              kernel_initializer=initializer,
                              bias_initializer=tf.zeros_initializer(),
                              kernel_regularizer=regularizer,
                              bias_regularizer=regularizer,
                              name='conv1')
        x9 = tf.layers.conv2d(x9, NUM_FEATURES*2, [3, 3], [1, 1], "SAME",
                              activation=None,
                              use_bias=True,
                              kernel_initializer=initializer,
                              bias_initializer=tf.zeros_initializer(),
                              kernel_regularizer=regularizer,
                              bias_regularizer=regularizer,
                              name='conv2')
        x9 = tf.layers.batch_normalization(x9, training=mode, name='bn')
        x9 = tf.nn.relu(x9, name='relu')

        print(x9)

    with tf.variable_scope('skip1'):

        # Upsampling size needs to match encoder2:
        x9up = tf.layers.conv2d_transpose(x9, NUM_FEATURES, [POOL_SIZE[0], POOL_SIZE[0]], [POOL_SIZE[0], POOL_SIZE[0]], "SAME",
                                        activation=None,
                                        use_bias=True,
                                        kernel_initializer=initializer,
                                        bias_initializer=tf.zeros_initializer(),
                                        kernel_regularizer=regularizer,
                                        bias_regularizer=regularizer,
                                        name='deconv')

        print(x9up)

        # Attention (x2, x9):
        a2 = tf.layers.conv2d(x2, NUM_FEATURES, [1, 1], [1, 1], "SAME",
                              activation=None,
                              use_bias=True,
                              kernel_initializer=initializer,
                              bias_initializer=tf.zeros_initializer(),
                              kernel_regularizer=regularizer,
                              bias_regularizer=regularizer,
                              name='attn1')
        a2 = tf.layers.batch_normalization(a2, training=mode, name='bn1')

        a9 = tf.layers.conv2d(x9up, NUM_FEATURES, [1, 1], [1, 1], "SAME",
                              activation=None,
                              use_bias=True,
                              kernel_initializer=initializer,
                              bias_initializer=tf.zeros_initializer(),
                              kernel_regularizer=regularizer,
                              bias_regularizer=regularizer,
                              name='attn2')
        a9 = tf.layers.batch_normalization(a9, training=mode, name='bn2')

        a29 = tf.nn.relu(a2 + a9, name='relu')
        a29 = tf.layers.conv2d(a29, 1, [1, 1], [1, 1], "SAME",
                               activation=None,
                               use_bias=True,
                               kernel_initializer=initializer,
                               bias_initializer=tf.zeros_initializer(),
                               kernel_regularizer=regularizer,
                               bias_regularizer=regularizer,
                               name='attn3')
        a29 = tf.layers.batch_normalization(a29, training=mode, name='bn3')
        a29 = tf.nn.sigmoid(a29, name='sigmoid')
        a29 = tf.multiply(a29, x2, name="multiply")

        print(a29)

        x10 = tf.concat([a29, x9up], axis=3, name='concat')  # encoder1 + skip1

    with tf.variable_scope('decoder1'):

        x10 = tf.layers.conv2d(x10, NUM_FEATURES, [3, 3], [1, 1], "SAME",
                              activation=None,
                              use_bias=True,
                              kernel_initializer=initializer,
                              bias_initializer=tf.zeros_initializer(),
                              kernel_regularizer=regularizer,
                              bias_regularizer=regularizer,
                              name='conv1')
        x10 = tf.layers.conv2d(x10, NUM_FEATURES, [3, 3], [1, 1], "SAME",
                              activation=None,
                              use_bias=True,
                              kernel_initializer=initializer,
                              bias_initializer=tf.zeros_initializer(),
                              kernel_regularizer=regularizer,
                              bias_regularizer=regularizer,
                              name='conv2')
        x10 = tf.layers.batch_normalization(x10, training=mode, name='bn')
        x10 = tf.nn.relu(x10, name='relu')

        print(x10)

    with tf.variable_scope('logits'):

        logits1 = tf.layers.conv2d(x10, 2, [1, 1], [1, 1], "SAME",
                                  activation=None,
                                  use_bias=True,
                                  kernel_initializer=initializer,
                                  bias_initializer=tf.zeros_initializer(),
                                  kernel_regularizer=regularizer,
                                  bias_regularizer=regularizer,
                                  name='conv1')
        logits2 = tf.layers.conv2d(x9, 2, [1, 1], [1, 1], "SAME",
                                   activation=None,
                                   use_bias=True,
                                   kernel_initializer=initializer,
                                   bias_initializer=tf.zeros_initializer(),
                                   kernel_regularizer=regularizer,
                                   bias_regularizer=regularizer,
                                   name='conv2')
        logits3 = tf.layers.conv2d(x8, 2, [1, 1], [1, 1], "SAME",
                                   activation=None,
                                   use_bias=True,
                                   kernel_initializer=initializer,
                                   bias_initializer=tf.zeros_initializer(),
                                   kernel_regularizer=regularizer,
                                   bias_regularizer=regularizer,
                                   name='conv3')
        logits4 = tf.layers.conv2d(x7, 2, [1, 1], [1, 1], "SAME",
                                   activation=None,
                                   use_bias=True,
                                   kernel_initializer=initializer,
                                   bias_initializer=tf.zeros_initializer(),
                                   kernel_regularizer=regularizer,
                                   bias_regularizer=regularizer,
                                   name='conv4')

        print(logits1)
        print(logits2)
        print(logits3)
        print(logits4)

        return logits1, logits2, logits3, logits4


def loss(logits1, logits2, logits3, logits4, labels1, labels2, labels3, labels4, loss_params):
    """
    logits = [BATCH_SIZE, NUM_ROWS, NUM_COLS, NUM_CLASSES]
    labels = [BATCH_SIZE, NUM_ROWS, NUM_COLS, 1]
    """

    # Cast to int32:
    labels1 = tf.cast(labels1[:, :, :, 0], tf.int32)
    labels2 = tf.cast(labels2[:, :, :, 0], tf.int32)
    labels3 = tf.cast(labels3[:, :, :, 0], tf.int32)
    labels4 = tf.cast(labels4[:, :, :, 0], tf.int32)

    # One-hot labels:
    one_hot_labels1 = tf.one_hot(labels1, 2)
    one_hot_labels2 = tf.one_hot(labels2, 2)
    one_hot_labels3 = tf.one_hot(labels3, 2)
    one_hot_labels4 = tf.one_hot(labels4, 2)

    # Probability:
    y_prob1 = tf.nn.softmax(logits1, 3)
    y_prob2 = tf.nn.softmax(logits2, 3)
    y_prob3 = tf.nn.softmax(logits3, 3)
    y_prob4 = tf.nn.softmax(logits4, 3)

    # Hausdorff loss:
    def get_hausdorff_conv_kernel(radius):
        xy = np.linspace(-radius, radius, (2 * radius) - 1)
        [x, y] = np.meshgrid(xy, xy)
        r = np.sqrt(x ** 2 + y ** 2)
        k = (r <= radius)
        n = np.sum(k.astype(int))
        kernel = k / n
        kernel = tf.convert_to_tensor(kernel[..., None, None], dtype=tf.float32)
        return kernel

    def soft_threshold(values, threshold=0.5):
        x = tf.abs(values) - threshold
        x_thresh = tf.multiply(tf.sign(values), tf.nn.relu(x))
        return x_thresh

    def hausdorff_loss(p, p_c, q, q_c, f_qp, f_pq, radius):
        Br = get_hausdorff_conv_kernel(radius)

        Cr1 = tf.nn.conv2d(p_c, Br, strides=[1, 1, 1, 1], padding='SAME')
        Tr1 = soft_threshold(Cr1, threshold=0.5)
        Or1 = tf.multiply(Tr1, f_qp)

        Cr2 = tf.nn.conv2d(p, Br, strides=[1, 1, 1, 1], padding='SAME')
        Tr2 = soft_threshold(Cr2, threshold=0.5)
        Or2 = tf.multiply(Tr2, f_pq)

        Cr3 = tf.nn.conv2d(q_c, Br, strides=[1, 1, 1, 1], padding='SAME')
        Tr3 = soft_threshold(Cr3, threshold=0.5)
        Or3 = tf.multiply(Tr3, f_pq)

        Cr4 = tf.nn.conv2d(q, Br, strides=[1, 1, 1, 1], padding='SAME')
        Tr4 = soft_threshold(Cr4, threshold=0.5)
        Or4 = tf.multiply(Tr4, f_qp)

        Lr = tf.reduce_mean(Or1 + Or2 + Or3 + Or4)
        return Lr

    # Scale 1:
    target1 = tf.expand_dims(tf.cast(labels1, tf.float32), axis=-1)
    output1 = tf.expand_dims(y_prob1[:, :, :, 1], axis=-1)
    p1 = target1
    p_c1 = 1 - p1
    q1 = tf.cast(output1 > 0.5, tf.float32)
    q_c1 = 1 - q1
    f_qp1 = tf.multiply(tf.square(target1 - output1), output1)
    f_pq1 = tf.multiply(tf.square(output1 - target1), target1)

    Lr3_1 = tf.pow(3.0, loss_params['hausdorff_power']) * hausdorff_loss(p1, p_c1, q1, q_c1, f_qp1, f_pq1, 3)
    Lr6_1 = tf.pow(6.0, loss_params['hausdorff_power']) * hausdorff_loss(p1, p_c1, q1, q_c1, f_qp1, f_pq1, 6)
    Lr9_1 = tf.pow(9.0, loss_params['hausdorff_power']) * hausdorff_loss(p1, p_c1, q1, q_c1, f_qp1, f_pq1, 9)
    Lr12_1 = tf.pow(12.0, loss_params['hausdorff_power']) * hausdorff_loss(p1, p_c1, q1, q_c1, f_qp1, f_pq1, 12)
    Lr15_1 = tf.pow(15.0, loss_params['hausdorff_power']) * hausdorff_loss(p1, p_c1, q1, q_c1, f_qp1, f_pq1, 15)
    Lr18_1 = tf.pow(18.0, loss_params['hausdorff_power']) * hausdorff_loss(p1, p_c1, q1, q_c1, f_qp1, f_pq1, 18)
    hausdorff_loss1 = Lr3_1 + Lr6_1 + Lr9_1 + Lr12_1 + Lr15_1 + Lr18_1

    # Scale 2:
    target2 = tf.expand_dims(tf.cast(labels2, tf.float32), axis=-1)
    output2 = tf.expand_dims(y_prob2[:, :, :, 1], axis=-1)
    p2 = target2
    p_c2 = 1 - p2
    q2 = tf.cast(output2 > 0.5, tf.float32)
    q_c2 = 1 - q2
    f_qp2 = tf.multiply(tf.square(target2 - output2), output2)
    f_pq2 = tf.multiply(tf.square(output2 - target2), target2)

    Lr3_2 = tf.pow(3.0, loss_params['hausdorff_power']) * hausdorff_loss(p2, p_c2, q2, q_c2, f_qp2, f_pq2, 3)
    Lr6_2 = tf.pow(6.0, loss_params['hausdorff_power']) * hausdorff_loss(p2, p_c2, q2, q_c2, f_qp2, f_pq2, 6)
    Lr9_2 = tf.pow(9.0, loss_params['hausdorff_power']) * hausdorff_loss(p2, p_c2, q2, q_c2, f_qp2, f_pq2, 9)
    Lr12_2 = tf.pow(12.0, loss_params['hausdorff_power']) * hausdorff_loss(p2, p_c2, q2, q_c2, f_qp2, f_pq2, 12)
    Lr15_2 = tf.pow(15.0, loss_params['hausdorff_power']) * hausdorff_loss(p2, p_c2, q2, q_c2, f_qp2, f_pq2, 15)
    Lr18_2 = tf.pow(18.0, loss_params['hausdorff_power']) * hausdorff_loss(p2, p_c2, q2, q_c2, f_qp2, f_pq2, 18)
    hausdorff_loss2 = Lr3_2 + Lr6_2 + Lr9_2 + Lr12_2 + Lr15_2 + Lr18_2

    # Scale 3:
    target3 = tf.expand_dims(tf.cast(labels3, tf.float32), axis=-1)
    output3 = tf.expand_dims(y_prob3[:, :, :, 1], axis=-1)
    p3 = target3
    p_c3 = 1 - p3
    q3 = tf.cast(output3 > 0.5, tf.float32)
    q_c3 = 1 - q3
    f_qp3 = tf.multiply(tf.square(target3 - output3), output3)
    f_pq3 = tf.multiply(tf.square(output3 - target3), target3)

    Lr3_3 = tf.pow(3.0, loss_params['hausdorff_power']) * hausdorff_loss(p3, p_c3, q3, q_c3, f_qp3, f_pq3, 3)
    Lr6_3 = tf.pow(6.0, loss_params['hausdorff_power']) * hausdorff_loss(p3, p_c3, q3, q_c3, f_qp3, f_pq3, 6)
    Lr9_3 = tf.pow(9.0, loss_params['hausdorff_power']) * hausdorff_loss(p3, p_c3, q3, q_c3, f_qp3, f_pq3, 9)
    Lr12_3 = tf.pow(12.0, loss_params['hausdorff_power']) * hausdorff_loss(p3, p_c3, q3, q_c3, f_qp3, f_pq3, 12)
    Lr15_3 = tf.pow(15.0, loss_params['hausdorff_power']) * hausdorff_loss(p3, p_c3, q3, q_c3, f_qp3, f_pq3, 15)
    Lr18_3 = tf.pow(18.0, loss_params['hausdorff_power']) * hausdorff_loss(p3, p_c3, q3, q_c3, f_qp3, f_pq3, 18)
    hausdorff_loss3 = Lr3_3 + Lr6_3 + Lr9_3 + Lr12_3 + Lr15_3 + Lr18_3

    # Scale 4:
    target4 = tf.expand_dims(tf.cast(labels4, tf.float32), axis=-1)
    output4 = tf.expand_dims(y_prob4[:, :, :, 1], axis=-1)
    p4 = target4
    p_c4 = 1 - p4
    q4 = tf.cast(output4 > 0.5, tf.float32)
    q_c4 = 1 - q4
    f_qp4 = tf.multiply(tf.square(target4 - output4), output4)
    f_pq4 = tf.multiply(tf.square(output4 - target4), target4)

    Lr3_4 = tf.pow(3.0, loss_params['hausdorff_power']) * hausdorff_loss(p4, p_c4, q4, q_c4, f_qp4, f_pq4, 3)
    Lr6_4 = tf.pow(6.0, loss_params['hausdorff_power']) * hausdorff_loss(p4, p_c4, q4, q_c4, f_qp4, f_pq4, 6)
    Lr9_4 = tf.pow(9.0, loss_params['hausdorff_power']) * hausdorff_loss(p4, p_c4, q4, q_c4, f_qp4, f_pq4, 9)
    Lr12_4 = tf.pow(12.0, loss_params['hausdorff_power']) * hausdorff_loss(p4, p_c4, q4, q_c4, f_qp4, f_pq4, 12)
    Lr15_4 = tf.pow(15.0, loss_params['hausdorff_power']) * hausdorff_loss(p4, p_c4, q4, q_c4, f_qp4, f_pq4, 15)
    Lr18_4 = tf.pow(18.0, loss_params['hausdorff_power']) * hausdorff_loss(p4, p_c4, q4, q_c4, f_qp4, f_pq4, 18)
    hausdorff_loss4 = Lr3_4 + Lr6_4 + Lr9_4 + Lr12_4 + Lr15_4 + Lr18_4

    # All scales:
    hausdorff_loss = hausdorff_loss1 + hausdorff_loss2 + hausdorff_loss3 + hausdorff_loss4

    # Dice loss:

    # Scale 1:
    y_overlap1 = tf.multiply(one_hot_labels1, y_prob1)
    intersection1 = tf.reduce_sum(y_overlap1, [1, 2])
    union_truth1 = tf.reduce_sum(tf.square(one_hot_labels1), [1, 2])
    union_pred1 = tf.reduce_sum(tf.square(y_prob1), [1, 2])
    dice_loss1 = 1 - tf.reduce_mean(2 * intersection1 / (union_truth1 + union_pred1))

    # Scale 2:
    y_overlap2 = tf.multiply(one_hot_labels2, y_prob2)
    intersection2 = tf.reduce_sum(y_overlap2, [1, 2])
    union_truth2 = tf.reduce_sum(tf.square(one_hot_labels2), [1, 2])
    union_pred2 = tf.reduce_sum(tf.square(y_prob2), [1, 2])
    dice_loss2 = 1 - tf.reduce_mean(2 * intersection2 / (union_truth2 + union_pred2))

    # Scale 3:
    y_overlap3 = tf.multiply(one_hot_labels3, y_prob3)
    intersection3 = tf.reduce_sum(y_overlap3, [1, 2])
    union_truth3 = tf.reduce_sum(tf.square(one_hot_labels3), [1, 2])
    union_pred3 = tf.reduce_sum(tf.square(y_prob3), [1, 2])
    dice_loss3 = 1 - tf.reduce_mean(2 * intersection3 / (union_truth3 + union_pred3))

    # Scale 4:
    y_overlap4 = tf.multiply(one_hot_labels4, y_prob4)
    intersection4 = tf.reduce_sum(y_overlap4, [1, 2])
    union_truth4 = tf.reduce_sum(tf.square(one_hot_labels4), [1, 2])
    union_pred4 = tf.reduce_sum(tf.square(y_prob4), [1, 2])
    dice_loss4 = 1 - tf.reduce_mean(2 * intersection4 / (union_truth4 + union_pred4))

    # All scales:
    dice_loss = dice_loss1 + dice_loss2 + dice_loss3 + dice_loss4

    # Hausdorff-Dice loss:
    hausdorff_dice_loss1 = hausdorff_loss1 + (hausdorff_loss1 / dice_loss1) * dice_loss1
    hausdorff_dice_loss2 = hausdorff_loss2 + (hausdorff_loss2 / dice_loss2) * dice_loss2
    hausdorff_dice_loss3 = hausdorff_loss3 + (hausdorff_loss3 / dice_loss3) * dice_loss3
    hausdorff_dice_loss4 = hausdorff_loss4 + (hausdorff_loss4 / dice_loss4) * dice_loss4
    hausdorff_dice_loss = hausdorff_dice_loss1 + hausdorff_dice_loss2 + hausdorff_dice_loss3 + hausdorff_dice_loss4

    # Regularization loss:
    reg_loss = tf.losses.get_regularization_loss()

    # Total loss:
    total_loss = hausdorff_dice_loss + reg_loss

    return hausdorff_loss, dice_loss, hausdorff_dice_loss, reg_loss, total_loss
