# -*- coding: utf-8 -*-
import tensorflow as tf


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.

        <Parameters>
        - sequence_length: 최대 문장 길이
        - num_classes: 클래스 개수
        - vocab_size: 등장 단어 수
        - embedding_size: 각 단어에 해당되는 임베디드 벡터의 차원
        - filter_sizes: convolutional filter들의 사이즈 (= 각 filter가 몇 개의 단어를 볼 것인가?) (예: "3, 4, 5")
        - num_filters: 각 filter size 별 filter 수
        - l2_reg_lambda: 각 weights, biases에 대한 l2 regularization 정도
    """

    def __init__(
            self, batch_normalization, activation_function, sequence_length, num_classes, vocab_size,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        chans_model = False

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        #self.prev_dropout_keep_prob = tf.placeholder(tf.float32, name="prev_dropout_keep_prob")
        self.phase_train = tf.placeholder(tf.bool, name='phase_train')

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        #with tf.device('/cpu:0'), tf.name_scope("embedding"):
        with tf.device('/gpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W_embed")
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        with tf.name_scope("input_dropout"):
            self.embedded_chars_expanded = tf.nn.dropout(self.embedded_chars_expanded , self.dropout_keep_prob)
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                #self.embedded_chars_expanded = tf.nn.dropout(self.embedded_chars_expanded , self.prev_dropout_keep_prob)
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                if chans_model:
                    conv = tf.nn.conv2d(
                        self.embedded_chars_expanded,
                        W,
                        strides=[1, 2, 2, 1],
                        padding="SAME",
                        name="conv")
                else:
                    conv = tf.nn.conv2d(
                        self.embedded_chars_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                if batch_normalization:
                    conv = self.batch_norm(conv, num_filters, self.phase_train)
                    #conv = tf.nn.local_response_normalization(conv, num_filters, self.phase_train)
                # Apply nonlinearity
                if activation_function == 'leaky_relu':
                    h = tf.nn.leaky_relu(tf.nn.bias_add(conv, b), alpha=0.01, name="leaky_relu")
                elif activation_function == 'elu':
                    h = tf.nn.elu(tf.nn.bias_add(conv, b), name="elu")
                elif activation_function == 'swish':
                    h = tf.nn.swish(tf.nn.bias_add(conv, b), name="swish")
                elif activation_function == 'relu':
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                else:
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                if chans_model:
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, 1, 1, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME',
                        name="pool1")
                   
                    nf2 = num_filters
                    aaa = int(round(embedding_size /4,0))
                    filter_shape1 = [filter_size, aaa , num_filters, nf2]
                    W1 = tf.Variable(tf.truncated_normal(filter_shape1, stddev=0.1), name="W1")
                    b1 = tf.Variable(tf.constant(0.1, shape=[nf2]), name="b1")
                    num_filters = nf2
                    conv1 = tf.nn.conv2d(
                        pooled,
                        W1,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv1")
                    conv1 = self.batch_norm(conv1, num_filters, self.phase_train)
                    h1 = tf.nn.elu(tf.nn.bias_add(conv1, b1), name="elu")
                    
                    pooled = tf.nn.max_pool(
                        h1,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool2")
                    
                    for i in range(0, 3):
                        pooled = tf.nn.conv2d(
                            pooled,
                            W1,
                            strides=[1, 1, 1, 1],
                            padding='SAME',
                            name="pool3_{}".format(i))
                        pooled = self.batch_norm(pooled, num_filters, self.phase_train)
                        pooled = tf.nn.elu(tf.nn.bias_add(pooled, b1), name="elu{}".format(i))
                else:
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")

                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
            #self.h_drop = tf.nn.dropout(self.h_drop, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            # W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.score_s = tf.nn.xw_plus_b(self.h_drop, W, b)

            if chans_model:
                aaa = tf.nn.dropout(self.score_s, self.dropout_keep_prob)
                W2 = tf.get_variable(
                    "W2",
                    shape=[num_classes, num_classes],
                    initializer=tf.contrib.layers.xavier_initializer())
                self.score_s = tf.nn.xw_plus_b(aaa, W2, b)

            self.scores = tf.nn.softmax(self.score_s, name="scores")
            self.predictions = tf.argmax(self.score_s, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.score_s, labels=self.input_y)
            #losses = tf.nn.sampled_softmax_loss(logits=self.score_s, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def batch_norm(self, x, n_out, phase_train):
        """
        Batch normalization on convolutional maps.
        Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
        Args:
            x:           Tensor, 4D BHWD input maps
            n_out:       integer, depth of input maps
            phase_train: boolean tf.Varialbe, true indicates training phase
            scope:       string, variable scope
        Return:
            normed:      batch-normalized maps
        """
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                           name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                            name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
        return normed
