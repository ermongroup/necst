# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Converts MNIST data to TFRecords file format with Example protos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys

import tensorflow as tf

from tensorflow.contrib.learn.python.learn.datasets import mnist
from scipy.io import loadmat

FLAGS = None


def _int64_feature(value):
  # TODO: for celebA, had to change value=[value] to value=value
  # for mnist, you may have to change this back to normal
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
  # return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(dataset, name):
  """Converts a dataset to tfrecords."""
  images = dataset.images
  # TODO: try this
  # images = images.astype(np.float32)
  # print(images.dtype)
  
  labels = dataset.labels
  num_examples = images.shape[0]

  if images.shape[0] != num_examples:
    raise ValueError('Images size %d does not match label size %d.' %
                     (images.shape[0], num_examples))

  filename = os.path.join(FLAGS.directory, FLAGS.dataset, name + '.tfrecords')
  print('Writing', filename)
  writer = tf.python_io.TFRecordWriter(filename)
  for index in range(num_examples):
    image_raw = images[index].tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'label': _int64_feature(labels[index]),
        'features': _bytes_feature(image_raw)}))
    writer.write(example.SerializeToString())
  writer.close()


def convert_stitched_mnist(dataset, name):
  """
  Converts a dataset to tfrecords.
  Specifically for this function, we are stitching 10 MNIST digits together
  to get a 7840-bit message

  There are currently no labels associated with these stitched digits
  """
  images = dataset.images
  num_examples = images.shape[0]
  filename = os.path.join(FLAGS.directory, FLAGS.dataset, name + '.tfrecords')
  print('Writing', filename)
  writer = tf.python_io.TFRecordWriter(filename)

  # stitch together 10 examples at a time
  for index in range(0, num_examples, 5):
    image_raw = images[index:(index+5)].tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'features': _bytes_feature(image_raw)}))
    writer.write(example.SerializeToString())
  writer.close()


# TODO: see if you want to incorporate the second label
# def convert_omniglot(dataset, name):
#   """Converts a dataset to tfrecords."""
#   images = dataset.images  
#   # images = images.astype(np.float32, copy=False)

#   labels = dataset.labels
#   num_examples = images.shape[0]

#   if images.shape[0] != num_examples:
#     raise ValueError('Images size %d does not match label size %d.' %
#                      (images.shape[0], num_examples))

#   filename = os.path.join(FLAGS.directory, FLAGS.dataset, name + '.tfrecords')
#   print('Writing', filename)
#   writer = tf.python_io.TFRecordWriter(filename)
#   for index in range(num_examples):
#     image_raw = images[index].tostring()
#     example = tf.train.Example(features=tf.train.Features(feature={
#         'label': _int64_feature(labels[index]),
#         'features': _bytes_feature(image_raw)
#         }))
#     writer.write(example.SerializeToString())
#   writer.close()


def convert_binary_mnist(fname, name):
  """Converts a dataset to tfrecords."""
  images = np.loadtxt(fname)
  images = images.astype(np.float32, copy=False)
  num_examples = len(images)

  filename = os.path.join(FLAGS.directory, FLAGS.dataset, name + '.tfrecords')
  print('Writing', filename)
  writer = tf.python_io.TFRecordWriter(filename)
  for index in range(num_examples):
    image_raw = images[index].tostring()
    # no labels given, so we leave those out here
    example = tf.train.Example(features=tf.train.Features(feature={
        'features': _bytes_feature(image_raw)}))
    writer.write(example.SerializeToString())
  writer.close()


def convert_random_bits(fname, name):
  """Converts a dataset to tfrecords."""
  images = np.load(fname)
  images = images.astype(np.float32, copy=False)
  num_examples = len(images)

  filename = os.path.join(FLAGS.directory, FLAGS.dataset, name + '.tfrecords')
  print('Writing', filename)
  writer = tf.python_io.TFRecordWriter(filename)
  for index in range(num_examples):
    image_raw = images[index].tostring()
    # no labels given, so we leave those out here
    example = tf.train.Example(features=tf.train.Features(feature={
        'features': _bytes_feature(image_raw)}))
    writer.write(example.SerializeToString())
  writer.close()


def main(unused_argv):
  # Get the data.

  if FLAGS.dataset == 'mnist':
    datasets = mnist.read_data_sets(FLAGS.directory,
                                     dtype=tf.uint8,
                                     reshape=False,
                                     validation_size=FLAGS.valid_size)

    # Convert to Examples and write the result to TFRecords.
    convert_to(datasets.train, 'mnist_train')
    convert_to(datasets.validation, 'mnist_valid')
    convert_to(datasets.test, 'mnist_test')
  elif FLAGS.dataset == 'BinaryMNIST':
    # assumes existence of .amat files in datasets directory
    # Convert to Examples and write the result to TFRecords
    convert_binary_mnist(os.path.join(FLAGS.directory, FLAGS.dataset,'binarized_mnist_train.amat'), 'bin_mnist_train')
    convert_binary_mnist(os.path.join(FLAGS.directory, FLAGS.dataset,'binarized_mnist_valid.amat'), 'bin_mnist_valid')
    convert_binary_mnist(os.path.join(FLAGS.directory, FLAGS.dataset,'binarized_mnist_test.amat'), 'bin_mnist_test')
  elif FLAGS.dataset == 'stitched_mnist':
    # assumes existence of .amat files in datasets directory
    # Convert to Examples and write the result to TFRecords
    # convert_stitched_mnist(os.path.join(FLAGS.directory, 'BinaryMNIST','binarized_mnist_train.amat'), 'stitched_bin_mnist_train')
    # convert_stitched_mnist(os.path.join(FLAGS.directory, 'BinaryMNIST','binarized_mnist_valid.amat'), 'stitched_bin_mnist_valid')
    # convert_stitched_mnist(os.path.join(FLAGS.directory, 'BinaryMNIST','binarized_mnist_test.amat'), 'stitched_bin_mnist_test')

    # this is for continuous grayscale MNIST 
    datasets = mnist.read_data_sets(FLAGS.directory,
                                   dtype=tf.uint8,
                                   reshape=False,
                                   validation_size=FLAGS.valid_size)

    # Convert to Examples and write the result to TFRecords.
    convert_stitched_mnist(datasets.train, 'five_stitched_mnist_train')
    convert_stitched_mnist(datasets.validation, 'five_stitched_mnist_valid')
    convert_stitched_mnist(datasets.test, 'five_stitched_mnist_test')
  elif FLAGS.dataset == 'random':
    # assumes existence of .npy files in datasets
    convert_random_bits(os.path.join(FLAGS.directory, FLAGS.dataset,'random_bits_train.npy'), 'rand_bits_train')
    convert_random_bits(os.path.join(FLAGS.directory, FLAGS.dataset,'random_bits_valid.npy'), 'rand_bits_valid')
    convert_random_bits(os.path.join(FLAGS.directory, FLAGS.dataset,'random_bits_test.npy'), 'rand_bits_test')
  elif FLAGS.dataset == 'omniglot':
    import h5py
    # file_path = os.path.join(FLAGS.directory, FLAGS.dataset, 'binary/' 'binary_omniglot.hdf5')
    file_path = os.path.join(FLAGS.directory, FLAGS.dataset, 'omniglot.hdf5')
    print(file_path)
    if not os.path.exists(file_path):
      raise ValueError('need: ', file_path)
    f = h5py.File(file_path, 'r')

    trainimages = f['train']
    validimages = f['valid']
    testimages = f['test']

    trainlabels = f['trainlabels']
    validlabels = f['validlabels']
    testlabels = f['testlabels']

    # convert one-hot encoded labels into integers
    print('converting one-hot encded class labels into integers...')
    trainlabels = np.reshape(np.argmax(trainlabels, axis=1), (-1, 1))
    validlabels = np.reshape(np.argmax(validlabels, axis=1), (-1, 1))
    testlabels = np.reshape(np.argmax(testlabels, axis=1), (-1, 1))

    # TODO: ignore these for now idk what they are
    trainlabels2 = f['trainlabels2']
    validlabels2 = f['validlabels2']
    testlabels2 = f['testlabels2']

    print(trainimages.shape, validimages.shape, testimages.shape)
    print(trainlabels.shape, validlabels.shape, testlabels.shape)
    print(trainlabels2.shape, validlabels2.shape, testlabels2.shape)

    from types import SimpleNamespace
    train_dataset = SimpleNamespace(images= trainimages, labels= trainlabels, labels2= trainlabels2)
    valid_dataset = SimpleNamespace(images= validimages, labels= validlabels, labels2= validlabels2)
    test_dataset = SimpleNamespace(images= testimages, labels= testlabels, labels2= testlabels2)

    # {'images': trainimages, 'labels': trainlabels, 'labels2': trainlabels2}
    # valid_dataset = {'images': validimages, 'labels': validlabels, 'labels2': validlabels2}
    # test_dataset = {'images': testimages, 'labels': testlabels, 'labels2': testlabels2}

    convert_to(train_dataset, 'omniglot_train')
    convert_to(valid_dataset, 'omniglot_valid')
    convert_to(test_dataset, 'omniglot_test')
  elif FLAGS.dataset == 'svhn':
    # load in the data
    file_path = os.path.join(FLAGS.directory, FLAGS.dataset, 'train_32x32.mat')
    print(file_path)
    train_set = loadmat(file_path)
    # train_set = loadmat(os.path.join(FLAGS.directory, FLAGS.dataset, 'train_32x32.mat'))
    trainimages = np.transpose(train_set['X'], (3, 0, 1, 2))
    trainlabels = train_set['y']

    # first combine extra training set with difficult examples and create validation set from this
    extra_set = loadmat(os.path.join(FLAGS.directory, FLAGS.dataset, 'extra_32x32.mat'))
    extraimages = np.transpose(extra_set['X'], (3, 0, 1, 2))
    extralabels = extra_set['y']

    # combine train and extra set
    combinedimages = np.concatenate((trainimages, extraimages))
    combinedlabels = np.concatenate((trainlabels, extralabels))

    # randomly select 10% validation set from this group
    n_valid = int(np.ceil(combinedlabels.shape[0] * 0.1))
    valid_idx = np.random.permutation(combinedlabels.shape[0])[0:n_valid]
    validimages = combinedimages[valid_idx]
    validlabels = combinedlabels[valid_idx]

    # subset of final training set
    train_idx = ~np.in1d(np.arange(combinedlabels.shape[0]), valid_idx)
    trainimages = combinedimages[train_idx]
    trainlabels = combinedlabels[train_idx]

    # load in test set
    test_set = loadmat(os.path.join(FLAGS.directory, FLAGS.dataset, 'test_32x32.mat'))
    testimages = np.transpose(test_set['X'], (3, 0, 1, 2))
    testlabels = test_set['y']

    # check shapes
    print(trainimages.shape, validimages.shape, testimages.shape)
    print(trainlabels.shape, validlabels.shape, testlabels.shape)

    from types import SimpleNamespace
    train_dataset = SimpleNamespace(images= trainimages, labels= trainlabels)
    extra_dataset = SimpleNamespace(images= extraimages, labels= extralabels)
    test_dataset = SimpleNamespace(images= testimages, labels= testlabels)

    convert_to(train_dataset, 'svhn_train')

    convert_to(extra_dataset, 'svhn_valid')
    convert_to(test_dataset, 'svhn_test')
  elif FLAGS.dataset == 'celebA':
    import h5py
    # file_path = os.path.join(FLAGS.directory, FLAGS.dataset, 'celeba_converted.hdf5')
    file_path = os.path.join(FLAGS.directory, FLAGS.dataset, 'REAL_gender_celeba_aligned_cropped.hdf5')
    # file_path = os.path.join(FLAGS.directory, FLAGS.dataset, 'REAL_celeba_aligned_cropped.hdf5')
    if not os.path.exists(file_path):
      raise ValueError('need: ', file_path)
    f = h5py.File(file_path, 'r')

    # retrieve images/labels
    trainimages = f['train']
    validimages = f['valid']
    testimages = f['test']

    trainlabels = f['trainlabels']
    validlabels = f['validlabels']
    testlabels = f['testlabels']

    # check shapes
    print(trainimages.shape, validimages.shape, testimages.shape)
    print(trainlabels.shape, validlabels.shape, testlabels.shape)

    from types import SimpleNamespace
    train_dataset = SimpleNamespace(images= trainimages, labels= trainlabels)
    valid_dataset = SimpleNamespace(images= validimages, labels= validlabels)
    test_dataset = SimpleNamespace(images= testimages, labels= testlabels)
    # train_dataset = SimpleNamespace(images= trainimages)
    # valid_dataset = SimpleNamespace(images= validimages)
    # test_dataset = SimpleNamespace(images= testimages)

    convert_to(train_dataset, 'gender_celebA_train')
    convert_to(valid_dataset, 'gender_celebA_valid')
    convert_to(test_dataset, 'gender_celebA_test')
  else:
    raise NotImplementedError


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--dataset',
      type=str,
      default='omniglot',
      help='Dataset (mnist/omniglot/BinaryMNIST/stitched_mnist/random/svhn/celebA)'
  )
  parser.add_argument(
      '--directory',
      type=str,
      default='../datasets',
      help='Directory to download data files and write the converted result'
  )
  parser.add_argument(
      '--valid_size',
      type=int,
      default=10000,
      help="""\
      Number of examples to separate from the training data for the validation
      set.\
      """
  )
  np.random.seed(1234)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
