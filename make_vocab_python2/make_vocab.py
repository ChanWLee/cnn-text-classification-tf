#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os

import tensorflow as tf

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
tf.flags.DEFINE_integer("embedding_dim", 16, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "2,3,4,5,6,7,8,9,10,11,12,13,14",
                       "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 16, "Number of filters per filter size (default: 128)")
# tf.flags.DEFINE_float("dropout_keep_prob", [0.4, 0.5, 0.6, 0.7], "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("dropout_keep_prob", [1.0], "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.001, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 5, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 200, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

data_loader = MultiClassDataLoader(tf.flags, WordDataProcessor())
data_loader.define_flags()

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

# Data Preparatopn
# ==================================================

# Load data
print("Loading data...")
x_train, y_train, x_dev, y_dev = data_loader.prepare_data()
vocab_processor = data_loader.vocab_processor

print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

vocab_processor.save(os.path.join(os.path.curdir, "vocab"))
