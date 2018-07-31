#!/usr/bin/python2

"""
A sample code for TensorFlow serving client in Python2.

Usage:
  root@1b02110513f8:/mycode/misc/tfserving_sample# python2 py_client/client.py --server 127.0.0.1:8500
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2 


tf.app.flags.DEFINE_string('server', 'localhost:8500',
                           'Server host:port.')
tf.app.flags.DEFINE_string('model', 'wide_and_deep',
                           'Model name.')
FLAGS = tf.app.flags.FLAGS


def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def main(_):
  host, port = FLAGS.server.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

  request = predict_pb2.PredictRequest()
  request.model_spec.name = FLAGS.model
  request.model_spec.signature_name = 'serving_default'

  feature_dict = {'age': _float_feature(value=40),
                  'capital_gain': _float_feature(value=0),
                  'capital_loss': _float_feature(value=0),
                  'education': _bytes_feature(value='Doctorate'.encode()),
                  'education_num': _float_feature(value=7),
                  'gender': _bytes_feature(value='Male'.encode()),
                  'hours_per_week': _float_feature(value=60),
                  'native_country': _bytes_feature(value='United-States'.encode()),
                  'occupation': _bytes_feature(value='Prof-specialty'.encode()),
                  'relationship': _bytes_feature(value='Own-child'.encode()),
                  'workclass': _bytes_feature(value='Private'.encode()),
                  }
  label = 1

  example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
  serialized = example.SerializeToString()

  request.inputs['inputs'].CopyFrom(
        tf.make_tensor_proto(serialized, shape=[1]))

  result_future = stub.Predict.future(request, 5.0)
  prediction = result_future.result().outputs['scores']

  print('True label: ' + str(label))
  print('Prediction: ' + str(np.argmax(prediction.float_val)))


if __name__ == '__main__':
  tf.app.run()
