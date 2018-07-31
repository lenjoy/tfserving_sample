#!/usr/bin/python3

"""
A sample code for TensorFlow training in Docker + Python3.

Training data can be downloaded from https://github.com/tensorflow/models/blob/master/official/wide_deep/census_dataset.py
It's from https://archive.ics.uci.edu/ml/machine-learning-databases/adult

Usage:
  * training in Docker:
  root@1b02110513f8:/mycode/misc/tfserving_sample# python3 train/train.py 

  * start server loading the model:
  root@1b02110513f8:/mycode/misc/tfserving_sample# tensorflow_model_server --port=8500 --model_name=wide_and_deep --model_base_path=/mycode/misc/census_model/serving_savemodel/ &

  * client:
  root@1b02110513f8:/mycode/misc/tfserving_sample# python2 client.py --server 127.0.0.1:8500

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import shutil
import sys

import tensorflow as tf  # pylint: disable=g-bad-import-order


_CSV_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket'
]

_CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
                        [0], [0], [0], [''], ['']]

_NUM_EXAMPLES = {
    'train': 32561,
    'validation': 16281,
}


LOSS_PREFIX = {'wide': 'linear/', 'deep': 'dnn/'}


def build_model_columns():
  """Builds a set of wide and deep feature columns."""
  # Continuous columns
  age = tf.feature_column.numeric_column('age')
  hours_per_week = tf.feature_column.numeric_column('hours_per_week')

  education = tf.feature_column.categorical_column_with_vocabulary_list(
      'education', [
          'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
          'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
          '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])

  # To show an example of hashing:
  occupation = tf.feature_column.categorical_column_with_hash_bucket(
      'occupation', hash_bucket_size=1000)

  # Transformations.
  age_buckets = tf.feature_column.bucketized_column(
      age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

  # Wide columns and deep columns.
  wide_columns = [
      education, occupation, age_buckets,
  ]

  deep_columns = [
      hours_per_week,
      tf.feature_column.indicator_column(education),
      tf.feature_column.indicator_column(age_buckets),

      # To show an example of embedding
      tf.feature_column.embedding_column(occupation, dimension=8),
  ]

  return wide_columns, deep_columns


def build_estimator(model_dir, wide_columns, deep_columns):
  """Build an estimator appropriate for the given model type."""
  hidden_units = [100, 75, 50, 25]

  # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
  # trains faster than GPU for this model.
  run_config = tf.estimator.RunConfig().replace(
      session_config=tf.ConfigProto(device_count={'GPU': 0}))

  return tf.estimator.DNNLinearCombinedClassifier(
      model_dir=model_dir,
      linear_feature_columns=wide_columns,
      dnn_feature_columns=deep_columns,
      dnn_hidden_units=hidden_units,
      config=run_config)


def input_fn(data_file, num_epochs, shuffle, batch_size):
    """Generate an input function for the Estimator."""

    assert tf.gfile.Exists(data_file), (
        '%s not found. Please make sure you have run data_download.py and '
        'set the --data_dir argument to the correct path.' % data_file)

#    table = tf.contrib.lookup.index_table_from_file(
#        vocabulary_file='test.txt', num_oov_buckets=1)

    def trans_tensor1(table, split_tags):
        tags_value = table.lookup(split_tags.values)
        categorial_tensor = tf.SparseTensor(
            indices=split_tags.indices,
            values=tags_value,
            dense_shape=split_tags.dense_shape)
        return categorial_tensor

    def trans_tensor2(table, split_tags):
        tags_value = table.lookup(split_tags.values)

        # Output: tags.indices Tensor("StringSplit:0", shape=(?, 2), dtype=int64)
        print('tags.indices', split_tags.indices)
        print('tags_value', tags_value)
        
        indice_idx = tf.map_fn(lambda x : x[0], split_tags.indices)
        print('indice_idx', indice_idx)
        value_idx = tf.map_fn(lambda x : x[1], split_tags.indices)
        print('value_idx', value_idx)
        
        value_arr = tf.cast(tf.gather(split_tags.values, value_idx), tf.int64)
        print('value_arr shape', value_arr.shape)

        new_indices = tf.stack([indice_idx, value_arr], axis=1)
        print('new_indices', new_indices)
        # new_values = [1 for x in range(value_arr.shape[0])]
        new_values = tf.ones_like(tags_value, tf.int64)
        print('new_values', new_values)

        print('split_tags shape', split_tags.get_shape())
        print('value_arr shape', value_arr.get_shape())
        print('new_indices shape', new_indices.get_shape())

        categorial_tensor = tf.SparseTensor(indices=new_indices,
                                            values=new_values,
                                            dense_shape=[new_indices.shape[1], 4])
        return categorial_tensor

    def parse_csv(value):
      print('Parsing', data_file)
      columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
      features = dict(zip(_CSV_COLUMNS, columns))
      labels = features.pop('income_bracket')
      return features, tf.equal(labels, '>50K')

    def parse_csv2(value):
        print('Parsing', data_file)
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, columns))

        # support multi-hot sparse features
        split_tags = tf.string_split([columns[6]], "-")
        print('tags.indices', split_tags.indices)

        categorial_tensor = trans_tensor1(table, split_tags)

#        with tf.Session() as s:
#            s.run([tf.global_variables_initializer(), tf.tables_initializer()])
#            print(s.run(split_tags))
        
        categorical_cols = {
            'occupation_sp': categorial_tensor}
        features.update(categorical_cols)
        
        labels = features.pop('income_bracket')
        return features, tf.equal(labels, '>50K')

    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TextLineDataset(data_file)

    # if shuffle:
    #     dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])

    dataset = dataset.map(parse_csv, num_parallel_calls=5)

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    return dataset


def dataset_input_fn(data_file, num_epochs, batch_size):
    dataset = tf.data.TFRecordDataset(data_file)

    # Use `tf.parse_single_example()` to extract data from a `tf.Example`
    # protocol buffer, and perform any additional per-record preprocessing.
    def parse_tfrecords(record):
        keys_to_features = {
            'age': tf.FixedLenFeature(
                (), tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
            'education_num': tf.FixedLenFeature(
                (), tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
            'capital_gain': tf.FixedLenFeature(
                (), tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
            'capital_loss': tf.FixedLenFeature(
                (), tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
            'hours_per_week': tf.FixedLenFeature(
                (), tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
            'education': tf.FixedLenFeature(
                (), tf.string),
            'marital_status': tf.FixedLenFeature(
                (), tf.string),
            'relationship': tf.FixedLenFeature(
                (), tf.string),
            'workclass': tf.FixedLenFeature(
                (), tf.string),
            'occupation': tf.FixedLenFeature((), tf.string),
            'occupation_sp': tf.VarLenFeature(tf.string),
            'income_bracket': tf.FixedLenFeature(
                (), tf.string),
        }
        parsed = tf.parse_single_example(record, keys_to_features)
        return {
            'age': parsed['age'],
            'education_num': parsed['education_num'],
            'capital_gain': parsed['capital_gain'],
            'capital_loss': parsed['capital_loss'],
            'hours_per_week': parsed['hours_per_week'],
            'education': parsed['education'],
            'marital_status': parsed['marital_status'],
            'relationship': parsed['relationship'],
            'workclass': parsed['workclass'],
            'occupation': parsed['occupation'],
            'occupation_sp': parsed['occupation_sp'],
               }, tf.equal(parsed['income_bracket'], '>50K')

    # Use `Dataset.map()` to build a pair of a feature dictionary and a label
    # tensor for each example.
    dataset = dataset.map(parse_tfrecords)
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)
    return dataset
    
    # iterator = dataset.make_one_shot_iterator()

    # `features` is a dictionary in which each value is a batch of values for
    # that feature; `labels` is a batch of labels.
    # features, labels = iterator.get_next()
    # return features, labels


def main():
    model_dir = '/mycode/misc/census_model'
    data_dir = '/mycode/misc/census_data'
    train_epochs = 4
    epochs_between_evals = 2
    batch_size = 40

    # Clean up the model directory if present
    shutil.rmtree(model_dir, ignore_errors=True)
    wide_columns, deep_columns = build_model_columns()
    model = build_estimator(model_dir, wide_columns, deep_columns)

    train_file = os.path.join(data_dir, 'adult.data')
    test_file = os.path.join(data_dir, 'adult.test')
    # train_file = os.path.join(data_dir, 'adult.data.tfrecords')
    # test_file = os.path.join(data_dir, 'adult.test.tfrecords')

    # Train and evaluate the model every `flags.epochs_between_evals` epochs.
    def train_input_fn():
        return input_fn(train_file, epochs_between_evals, True, batch_size)
        # return dataset_input_fn(train_file, epochs_between_evals, batch_size)

    def eval_input_fn():
        return input_fn(test_file, 1, False, batch_size)
        # return dataset_input_fn(test_file, epochs_between_evals, batch_size)
    
    # Train and evaluate the model every `flags.epochs_between_evals` epochs.
    for n in range(train_epochs // epochs_between_evals):
        model.train(input_fn=train_input_fn)
        results = model.evaluate(input_fn=eval_input_fn)

        # Display evaluation metrics
        print('========== Results at epoch', (n + 1) * epochs_between_evals)
        print('-' * 60)

        for key in sorted(results):
            print('%s: %s' % (key, results[key]))

    feature_columns = wide_columns + deep_columns
    feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
    serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)

    servable_model_dir = model_dir + '/serving_savemodel'
    servable_model_path = model.export_savedmodel(servable_model_dir, serving_input_receiver_fn)
    print('Saved to {}'.format(servable_model_path))


if __name__ == '__main__':
  main()
