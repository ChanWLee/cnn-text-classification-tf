import tensorflow as tf
from tensorflow.contrib import rnn

class TextRNN(object):

    def __init__(self, batch_normalization, sequence_length, num_classes, vocab_size,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        embedding_size =2
        print('sl={}, nc={},vx={},es={},fs={},nf={}'.format(sequence_length, num_classes, vocab_size,embedding_size, len(filter_sizes), num_filters))

        hidden_dim = num_filters
        batch_size = 64
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.phase_train = tf.placeholder(tf.bool, name='phase_train')


        with tf.device('/cpu:0'), tf.name_scope("embedding"):

            W = tf.get_variable(
                "W",
                shape=[hidden_dim, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            w = tf.get_variable("w",[hidden_dim, num_classes])
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        
        def lstm_cell():
            lstm = rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True)
            #return lstm
            return tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=self.dropout_keep_prob)

        multi_cells = rnn.MultiRNNCell([lstm_cell() for _ in range(embedding_size)], state_is_tuple=True)
        initial_state = multi_cells.zero_state(batch_size, tf.float32)
        
        rnn_inputs = [tf.squeeze(i, axis=[1]) for i in tf.split(self.embedded_chars, sequence_length, 1)]

        def loop(prev, _):
            prev = tf.matmul(prev, W) + b
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(W, prev_symbol)

        #rnn_inputs = [i for i in tf.split(x_one_hot, sequence_length, 1)]
        #outputs, _state = tf.contrib.rnn.static_rnn(multi_cells, rnn_inputs, initial_state=initial_state )
        #outputs, _state = tf.nn.dynamic_rnn(multi_cells, rnn_inputs, dtype=tf.float32)
        outputs, _state = tf.contrib.legacy_seq2seq.rnn_decoder(rnn_inputs, initial_state, multi_cells
                #,loop_function=loop
                ,scope='rnnlm')

        seq_output = tf.concat(outputs, axis=1)
        #output = tf.reshape(seq_output, [-1, hidden_dim])
        output = seq_output

        with tf.name_scope("output"):
            W1 = tf.get_variable(
                "W1",
                shape=[num_filters*sequence_length, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            #b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            preds = self.scores = tf.nn.xw_plus_b(output, W1, b, name="scores")
            #logits = tf.matmul(output, softmax_w) + softmax_b
            #preds = tf.nn.softmax(logits)
            self.predictions = tf.argmax(preds, 1, name="predictions")


        with tf.name_scope("loss"):
            y_reshaped = tf.reshape(self.input_y, [-1, num_classes])
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)

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
