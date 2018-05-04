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

        # x,y one_hot encoding
        x_one_hot = tf.one_hot(self.input_x, num_classes)

        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable("softmax_w", [hidden_dim, vocab_size])
            sotfmax_b = tf.get_variable("sotfmax_b", [vocab_size])

        lstm = rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True)
        def lstm_cell():
            return tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=self.dropout_keep_prob)

        multi_cells = rnn.MultiRNNCell([lstm_cell() for _ in range(len(filter_sizes))], state_is_tuple=True)
        #initial_state = multi_cells.zero_state(filter_sizes, tf.float32)

        rnn_inputs = [tf.squeeze(i, axis=[1]) for i in tf.split(x_one_hot, sequence_length, 1)]

        #outputs, _state = tf.contrib.rnn.static_rnn(multi_cells, rnn_inputs )
        outputs, _state = tf.nn.dynamic_rnn(multi_cells, rnn_inputs, dtype=tf.float32)

        seq_output = tf.concat(outputs, axis=1)
        output = tf.reshape(seq_output, [-1, hidden_dim])

        with tf.variable_scope('softmax'):
            softmax_w = tf.Variable(tf.truncated_normal((hidden_dim, num_classes), stddev=0.1))
            sotfmax_b = tf.Variable(tf.zeros(num_classes))

        with tf.name_scope("output"):
            logits = tf.matmul(output, softmax_w) + softmax_b
            self.predictions = tf.nn.softmax(logits, name='predictions')


        with tf.name_scope("loss"):
            y_reshaped = tf.reshape(self.input_y, [-1, num_classes])
            losses = tf.nn.sotfmax_cross_entropy_with_logits_v2(logits=logits, labels=y_reshaped)
            self.loss = tf.reduce_mean(loss)

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

'''
        em = tf.get_variable("embedding", [vocab_size, hidden_dim])
        self.embedded_chars = tf.nn.embedding_lookup(em, self.input_x)
        inputs = tf.split(self.embedded_chars, sequence_length, 1)
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        with tf.name_scope("output"):
            def loop(prev, _):
                prev = tf.matmul(pre, softmax_w) + softmax_b
                prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
                return tf.nn.embedding_lookup(em, prev_symbol)
            #outputs, _states = tf.nn.dynamic_rnn(multi_cells, self.embedded_chars, dtype=tf.float32)
            #outputs, _states = tf.nn.dynamic_rnn(multi_cells, inputs, dtype=tf.float32)
            #self.output = tf.contrib.layers.fully_connected(outputs[:, -1], num_classes, activation_fn=None)
            outputs, _s = tf.contrib.legacy_seq2seq.rnn_decoder(inputs, initial_state, multi_cells, loop_function=loop, scope='rnnlm')
            self.output = tf.reshape(tf.concat(outputs, 1), [-1, num_classes])

            #self.h_drop = tf.nn.dropout(self.flat, self.dropout_keep_prob)
            #self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.output, 1, name="predictions")

        with tf.name_scope("loss"):
            self.loss = tf.reduce_sum(tf.square(self.output- self.input_y))
            #losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.input_y)
            #self.loss = tf.reduce_mean(losses)

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        #train = tf.train.RMSPropOptimizer(learing_rate).minimize(loss)
'''
