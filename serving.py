#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np
import tensorflow as tf

import data_helpers
from multi_class_data_loader import MultiClassDataLoader
from word_data_processor import WordDataProcessor

tf.flags.DEFINE_integer("batch_size", 256, "Batch Size (default: 64)")
# tf.flags.DEFINE_string("checkpoint_dir", "./runs/1509332332/checkpoints/", "")
# tf.flags.DEFINE_string("checkpoint_dir", "./runs/1510118340/checkpoints/", "")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_boolean("model_version", 1, "")

data_loader = MultiClassDataLoader(tf.flags, WordDataProcessor())
data_loader.define_flags()

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
# print("\nParameters:")
# for attr, value in sorted(FLAGS.__flags.items()):
#     print("{}={}".format(attr.upper(), value))
# print("")
#
# if FLAGS.eval_train:
#     x_raw, y_test = data_loader.load_data_and_labels()
#     y_test = np.argmax(y_test, axis=1)
# else:
#     x_raw, y_test = data_loader.load_dev_data_and_labels()
#     y_test = np.argmax(y_test, axis=1)

# checkpoint_dir이 없다면 가장 최근 dir 추출하여 셋팅
if FLAGS.checkpoint_dir == "":
    all_subdirs = ["./runs/" + d for d in os.listdir('./runs/.') if os.path.isdir("./runs/" + d)]
    latest_subdir = max(all_subdirs, key=os.path.getmtime)
    FLAGS.sub_dir = latest_subdir
    FLAGS.checkpoint_dir = latest_subdir + "/checkpoints/"

# # Map data into vocabulary
# vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
# vocab_processor = data_loader.restore_vocab_processor(vocab_path)
# x_test = np.array(list(vocab_processor.transform(x_raw)))

# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        export_path = os.path.abspath(os.path.join(FLAGS.sub_dir, "serving", str(FLAGS.model_version)))
        print('Exporting trained model to \n', export_path)
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)

        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        input_x = graph.get_operation_by_name("input_x").outputs[0]
        input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        phase_train = graph.get_operation_by_name("phase_train").outputs[0]
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # input_rawx = tf.placeholder(tf.string, shape=([],68))


        # classify_inputs_x = tf.saved_model.utils.build_tensor_info(
        #     input_x
        # )
        # classify_input_dropout = tf.saved_model.utils.build_tensor_info(
        #     dropout_keep_prob
        # )
        # classify_input_phase_train = tf.saved_model.utils.build_tensor_info(
        #     phase_train
        # )
        # prediction_output_tensor_info = tf.saved_model.utils.build_tensor_info(
        #     predictions
        # )
        #
        # classification_signature = (
        #     tf.saved_model.signature_def_utils.build_signature_def(
        #         inputs={
        #             tf.saved_model.signature_constants.CLASSIFY_INPUTS_X:
        #                 classify_inputs_x,
        #             tf.saved_model.signature_constants.CLASSIFY_INPUT_DROPOUT:
        #                 classify_input_dropout,
        #             tf.saved_model.signature_constants.CLASSIFY_INPUT_PHASE_TRAIN:
        #                 classify_input_phase_train
        #         },
        #         outputs={
        #             # tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES:
        #             #     classification_outputs_classes,
        #             tf.saved_model.signature_constants.PREDICTION_OUTPUT_TENSOR_INFO:
        #                 prediction_output_tensor_info
        #         },
        #         method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME))

        tensor_info_x = tf.saved_model.utils.build_tensor_info(input_x)
        # tensor_info_rawx = tf.saved_model.utils.build_tensor_info(input_rawx)
        tensor_info_dropout = tf.saved_model.utils.build_tensor_info(dropout_keep_prob)
        tensor_info_prediction = tf.saved_model.utils.build_tensor_info(predictions)
        tensor_info_phase_train = tf.saved_model.utils.build_tensor_info(phase_train)

        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'inputx': tensor_info_x,
                        'dropout': tensor_info_dropout,
                        'phase_train': tensor_info_phase_train},
                outputs={
                    # 'rawx': tensor_info_rawx,
                         'prediction': tensor_info_prediction},
                # method_name="positive_negative"))
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

        legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'predict_sentiment_score':
                    prediction_signature,
                # tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                #     classification_signature,
            },
            legacy_init_op=legacy_init_op)

        builder.save()

        print("Done exporting!")
