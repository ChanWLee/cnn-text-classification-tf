import tensorflow as tf
from tensorflow.contrib import rnn

class TextRNN(object):

    def __init__(self, batch_normalization, sequence_length, num_classes, vocab_size,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        print('sl={}, nc={},vx={},es={},fs={},nf={}'.format(sequence_length, num_classes, vocab_size,embedding_size, len(filter_sizes), num_filters))

        RNNs = ['static_rnn', 'dynamic_rnn', 'bidirectional_rnn', 'seq2seq']
        RNN = RNNs[2]
        filter_size = len(filter_sizes)
        #hidden_dim = embedding_size
        hidden_dim = num_filters
        batch_size = 64
        time_step_size = vocab_size
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        #self.input_x = tf.placeholder(tf.int32, [None, sequence_length, hidden_dim], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.phase_train = tf.placeholder(tf.bool, name='phase_train')


        b = tf.Variable(tf.constant(0.1, shape=[num_filters]))

        l2_loss = tf.constant(0.0)
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            '''
            W2= tf.get_variable(
                "W2",
                shape=[hidden_dim, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            '''
            W = tf.Variable(
                tf.random_uniform([vocab_size, filter_size], -1.0, 1.0),
                name="W")
            '''
            xt = tf.transpose(self.input_x, [1,0])
            xr = tf.reshape(xt, [-1, hidden_dim])
            x_split = tf.split(xr, time_step_size, 0)

            w = tf.get_variable("w",[hidden_dim, num_classes])
            '''
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)



        def cell():
            lstm = rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True)
            #return lstm
            return tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=self.dropout_keep_prob)

        cells = rnn.MultiRNNCell([cell() for _ in range(filter_size)], state_is_tuple=True)
        #initial_state = cells.zero_state(batch_size, tf.float32)

        rnn_inputs = [tf.squeeze(i, axis=[1]) for i in tf.split(self.embedded_chars, sequence_length, 1)]
        #rnn_inputs = x_split

        #outputs, _state = tf.contrib.rnn.static_rnn(cells, rnn_inputs, initial_state=initial_state )
        all = hidden_dim * sequence_length
        if RNN == 'static_rnn':
            outputs, _state = tf.contrib.rnn.static_rnn(cells, rnn_inputs, dtype=tf.float32)
        elif RNN == 'bidirectional_rnn':
            #rnn_inputs = [tf.squeeze(i, axis=[1]) for i in tf.split(self.input_x, hidden_dim, 1)]
            rnn_inputs = self.input_x
            rnn_inputs = self.embedded_chars_expanded
            rnn_inputs = tf.reshape(rnn_inputs, [-1, sequence_length, 8])
            rnn_inputs = tf.cast(rnn_inputs, tf.float32)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell(), cell_bw=cell(),
                inputs=rnn_inputs, dtype=tf.float32
                #inputs=tf.cast(self.input_x,tf.float32), dtype=tf.float32
                )
            #fw = tf.transpose(outputs[0], [1,0,2])
            #bw = tf.transpose(outputs[1], [1,0,2])
            fw = tf.transpose(outputs[0], [2,0,1])
            bw = tf.transpose(outputs[1], [2,0,1])
            outputs = [fw[-1], bw[-1]]
            #all = filter_size * hidden_dim
            all = sequence_length
        elif RNN == 'dynamic_rnn':
            rnn_inputs = [tf.squeeze(i, axis=[1]) for i in tf.split(self.embedded_chars_expanded, sequence_length, 1)]
            outputs, _state = tf.nn.dynamic_rnn(cells, rnn_inputs, dtype=tf.float32)
        elif RNN == 'seq2seq':
            outputs, _state = tf.contrib.legacy_seq2seq.rnn_decoder(rnn_inputs, initial_state, cells
                #,loop_function=loop
                ,scope='rnnlm')

        seq_output = tf.concat(outputs, axis=1)
        output = tf.reshape(seq_output, [-1, all])
        #output = seq_output
        print('----------------')
        print(output.shape)

        with tf.name_scope("output"):
            #all = hidden_dim*sequence_length
            W1 = tf.get_variable(
                "W1",
                shape=[all, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]))
            l2_loss += tf.nn.l2_loss(W1)
            l2_loss += tf.nn.l2_loss(b)
            preds = self.scores = tf.nn.xw_plus_b(output, W1, b, name="scores")
            print('score:{}'.format(preds))
            #logits = tf.matmul(output, softmax_w) + softmax_b
            #preds = tf.nn.softmax(logits)
            self.predictions = tf.argmax(preds, 1, name="predictions")


        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_loss

        with tf.name_scope("accuracy"):
            print('predictions:{}\n    input_y:{}'.format(self.predictions, self.input_y))
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
            #outputs, _states = tf.nn.dynamic_rnn(cells, self.embedded_chars, dtype=tf.float32)
            #outputs, _states = tf.nn.dynamic_rnn(cells, inputs, dtype=tf.float32)
            #self.output = tf.contrib.layers.fully_connected(outputs[:, -1], num_classes, activation_fn=None)
            outputs, _s = tf.contrib.legacy_seq2seq.rnn_decoder(inputs, initial_state, cells, loop_function=loop, scope='rnnlm')
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
