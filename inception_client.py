# -*- coding: utf-8 -*-

from __future__ import print_function

import tensorflow as tf
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow.contrib import learn
import numpy as np

# This is a placeholder for a Google-internal import.

tf.app.flags.DEFINE_string('server', 'localhost:9000',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('text', '학교에 가기 싫어요', 'text')
tf.app.flags.DEFINE_string('file', '', 'path to text file')
tf.app.flags.DEFINE_string('vocab', '', 'path to vocab file')
FLAGS = tf.app.flags.FLAGS


def transform_vocab(vocab_path, data):
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    return np.array(list(vocab_processor.transform(data)))

def main(_):
    host, port = FLAGS.server.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    if FLAGS.file:
        with open(FLAGS.file, 'rb') as f:
            data = []
            data_list = []
            for line in f.readlines():
                data += line.strip().split(',')
                data_list.append(line)
            x_test = data
            if FLAGS.vocab:
                x_test = transform_vocab(FLAGS.vocab, data)
            tf_data = [int(i) for i in x_test]
            leng_data = len(data_list)

            request = predict_pb2.PredictRequest()
            request.model_spec.name = 'inception'
            request.model_spec.signature_name = 'predict_sentiment_score'
            # request.inputs['rawx'].CopyFrom(
            #     tf.contrib.util.make_tensor_proto(data, shape=[leng_data, 68]))
            request.inputs['inputx'].CopyFrom(
                tf.contrib.util.make_tensor_proto(tf_data, shape=[leng_data, 68]))
            request.inputs['dropout'].CopyFrom(
                tf.contrib.util.make_tensor_proto(1.0, shape=[1]))
            request.inputs['phase_train'].CopyFrom(
                tf.contrib.util.make_tensor_proto(False, dtype=tf.bool))
            result = stub.Predict(request, 10.0)  # 10 secs timeout
            print(result)

if __name__ == '__main__':
    tf.app.run()
    # main()
