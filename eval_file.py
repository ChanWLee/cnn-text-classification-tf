#! /usr/bin/env python
# -*- coding: utf-8 -*-
import json

import tensorflow as tf
import numpy as np
import os
import data_helpers
from multi_class_data_loader import MultiClassDataLoader
from word_data_processor import WordDataProcessor
import csv

# Parameters
# ==================================================

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 256, "Batch Size (default: 64)")
# tf.flags.DEFINE_string("checkpoint_dir", "./runs/1509332332/checkpoints/", "")
# tf.flags.DEFINE_string("checkpoint_dir", "./runs/1510118340/checkpoints/", "")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

data_loader = MultiClassDataLoader(tf.flags, WordDataProcessor())
data_loader.define_flags()

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

if FLAGS.eval_train:
    x_raw, y_test = data_loader.load_data_and_labels()
    y_test = np.argmax(y_test, axis=1)
else:
    x_raw, y_test = data_loader.load_dev_data_and_labels()
    y_test = np.argmax(y_test, axis=1)

# checkpoint_dir이 없다면 가장 최근 dir 추출하여 셋팅
if FLAGS.checkpoint_dir == "":
    all_subdirs = ["./runs/" + d for d in os.listdir('./runs/.') if os.path.isdir("./runs/" + d)]
    latest_subdir = max(all_subdirs, key=os.path.getmtime)
    FLAGS.checkpoint_dir = latest_subdir + "/checkpoints/"

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = data_loader.restore_vocab_processor(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))

out_path = os.path.join(FLAGS.checkpoint_dir, "../", "x_test")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(x_test)