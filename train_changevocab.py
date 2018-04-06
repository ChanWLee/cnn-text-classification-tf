#! /usr/bin/env python
# -*- coding: utf-8 -*-
import random

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from multi_class_data_loader import MultiClassDataLoader
from word_data_processor import WordDataProcessor

# Parameters
# ==================================================

# Model Hyperparameters
"""
    <Parameters>
        - embedding_dim: 각 단어에 해당되는 임베디드 벡터의 차원,  차원축소가필요( 일정차원을 넘으면 성능떨어짐)
        - filter_sizes: convolutional filter들의 사이즈 (= 각 filter가 몇 개의 단어를 볼 것인가?) (예: "3, 4, 5")
        - num_filters: 각 filter size 별 filter 수
        - l2_reg_lambda: 각 weights, biases에 대한 l2 regularization 정도중요( 0.001 기본, 아주중요 0.1)
        - batch_size: 몇개의 데이터셋을 가지고 와서 웨이트를 업데이트 (배치사이즈1,데이터셋1만, 1개씩쪼개서1만번)
        - num_epochs: 데이터셋을 한번 다 돌면 epoch 1
        
        - sequence_length: 최대 문장 길이
        - num_classes: 클래스 개수
        - vocab_size: 등장 단어 수
"""
tf.flags.DEFINE_integer("embedding_dim", 64, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
# tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 192, "Number of filters per filter size (default: 128)")
#tf.flags.DEFINE_float("dropout_keep_prob", [0.3, 0.4, 0.5, 0.6, 0.7], "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.4, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.1, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 1000, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 20, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

data_loader = MultiClassDataLoader(tf.flags, WordDataProcessor())
data_loader.define_flags()

FLAGS = tf.flags.FLAGS
FLAGS.mark_as_parsed()
print("\nParameters:")
p_param = list()
for attr, value in sorted(FLAGS.flag_values_dict().items()):
    print("{}={}".format(attr.upper(), value))
    p_param.append('{}={}'.format(attr.upper(), value))
print("")

# Data Preparatopn
# ==================================================

# Load data
time_str = datetime.datetime.now().isoformat()
print("{}\nLoading data...".format(time_str))
vocab_path = os.path.join("./", "", "vocab")

try:
    print("Load prev vocab...")
    vocab_processor = data_loader.restore_vocab_processor(vocab_path)

    # using prev vocab
    print("load train, dev data...{}".format(datetime.datetime.now().isoformat()))
    x_train, y_train = data_loader.load_train_data_and_labels()
    x_dev, y_dev = data_loader.load_dev_data_and_labels()
    #vocab_processor = data_loader.restore_vocab_processor(vocab_path)

    print("transform train, dev data by vocab...{}".format(datetime.datetime.now().isoformat()))
    x_train = np.array(list(vocab_processor.transform(x_train)))
    x_dev = np.array(list(vocab_processor.transform(x_dev)))
    #y_train = np.argmax(y_train, axis=1)
    #y_dev = np.argmax(y_dev, axis=1)

except Exception as e:
    print("New vocab...")
    # new vocab
    x_train, y_train, x_dev, y_dev = data_loader.prepare_data()
    vocab_processor = data_loader.vocab_processor

time_str = datetime.datetime.now().isoformat()
print("{}: Finish loading data".format(time_str))
print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))


# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)

    with sess.as_default():
        cnn = TextCNN(
            batch_normalization=False,
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)#default
        #optimizer = tf.train.RMSPropOptimizer(1e-3, 0.9)
        # optimizer = tf.train.AdamOptimizer(3e-2)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        #timestamp = '1522347132'
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Write parameter
        f_param = open(os.path.join(out_dir,"param"), 'w')
        f_param.write('\n'.join(p_param))
        f_param.close()

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            #random_dropout = random.choice(FLAGS.dropout_keep_prob)
            random_dropout = FLAGS.dropout_keep_prob
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: random_dropout,
              cnn.phase_train: True
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)

            # time_str = datetime.datetime.now().isoformat()
            # print("{}: step {}, dropout {}, loss {:g}, acc {:g}".format(time_str, step, random_dropout, loss, accuracy))
            if step % (FLAGS.evaluate_every/5) == 0:
                train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0,
              cnn.phase_train: False
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                 writer.add_summary(summaries, step)

        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        for batch in batches:
            try:
                x_batch, y_batch = zip(*batch)
            except:
                pass
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                # print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                # print("Saved model checkpoint to {}\n".format(path))
        # serving()
