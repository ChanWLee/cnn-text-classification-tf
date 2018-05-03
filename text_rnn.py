import tensorflow as tf
from tensorflow.contrib import rnn

class TextRNN(object):

    def __init__(self, batch_normalization, sequence_length, num_classes, vocab_size,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        hidden_dim = num_filters
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.phase_train = tf.placeholder(tf.bool, name='phase_train')

        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        def lstm_cell():
            return rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True)

        cell = rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True)
        multi_cells = rnn.MultiRNNCell([lstm_cell() for _ in range(len(filter_sizes))], state_is_tuple=True)

        with tf.name_scope("output"):
            outputs, _states = tf.nn.dynamic_rnn(multi_cells, self.embedded_chars, dtype=tf.float32)
            self.predictions = tf.contrib.layers.fully_connected(outputs[:, -1], num_classes, activation_fn=None)

        with tf.name_scope("loss"):
            self.loss = tf.reduce_sum(tf.square(self.predictions - self.input_y))
            #losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.input_y)
            #self.loss = tf.reduce_mean(losses)

        with tf.name_scope("accuracy"):
            self.predictions = tf.argmax(self.predictions, 1)
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        #train = tf.train.RMSPropOptimizer(learing_rate).minimize(loss)
