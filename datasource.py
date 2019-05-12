"""
Code for loading data. 
lifted mostly from aditya-grover's UAE project
"""
import numpy as np
import random
import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.python.platform import flags
FLAGS = flags.FLAGS

class Datasource(object):

	def __init__(self, sess):

		self.sess = sess
		self.seed = FLAGS.seed
		tf.set_random_seed(self.seed)
		np.random.seed(self.seed)

		self.batch_size = FLAGS.batch_size

		# minor changes
		if FLAGS.datasource == 'mnist' or FLAGS.datasource == 'omniglot2mnist':

			self.target_dataset = 'mnist'
			self.TRAIN_FILE = 'mnist_train.tfrecords'
			self.VALID_FILE = 'mnist_valid.tfrecords'
			self.TEST_FILE = 'mnist_test.tfrecords'

			self.input_dim = 784
			self.num_classes = 10
			self.dtype = tf.float32
			self.preprocess = self._preprocess_mnist
			self.get_dataset = self.get_tf_dataset

		elif FLAGS.datasource == 'BinaryMNIST':

			# added BinaryMNIST for testing purposes
			self.target_dataset = 'BinaryMNIST'
			self.TRAIN_FILE = 'bin_mnist_train.tfrecords'
			self.VALID_FILE = 'bin_mnist_valid.tfrecords'
			self.TEST_FILE = 'bin_mnist_test.tfrecords'

			self.input_dim = 784
			self.num_classes = 10
			self.dtype = tf.float32
			self.preprocess = self._preprocess_binary_mnist
			self.get_dataset = self.get_binary_tf_dataset		

		elif FLAGS.datasource == 'random':

			# added BinaryMNIST for testing purposes
			self.target_dataset = 'random'
			self.TRAIN_FILE = 'rand_bits_train.tfrecords'
			self.VALID_FILE = 'rand_bits_valid.tfrecords'
			self.TEST_FILE = 'rand_bits_test.tfrecords'

			self.input_dim = 100
			self.num_classes = 10
			self.dtype = tf.float32
			# we can re-use the functions for BinaryMNIST since these also don't have class labels
			self.preprocess = self._preprocess_binary_mnist
			self.get_dataset = self.get_binary_tf_dataset

		elif FLAGS.datasource == 'omniglot':

			self.target_dataset = 'omniglot'
			self.TRAIN_FILE = 'omniglot_train.tfrecords'
			self.VALID_FILE = 'omniglot_valid.tfrecords'
			self.TEST_FILE = 'omniglot_test.tfrecords'

			self.input_dim = 784
			self.num_classes = 50
			self.dtype = tf.float32
			self.preprocess = self._preprocess_omniglot
			self.get_dataset = self.get_tf_dataset

		elif FLAGS.datasource == 'binary_omniglot':

			self.target_dataset = 'binary_omniglot'
			self.TRAIN_FILE = 'binary_omniglot_train.tfrecords'
			self.VALID_FILE = 'binary_omniglot_valid.tfrecords'
			self.TEST_FILE = 'binary_omniglot_test.tfrecords'

			self.input_dim = 784
			self.num_classes = 50
			self.dtype = tf.float32
			self.preprocess = self._preprocess_omniglot
			self.get_dataset = self.get_tf_dataset

		elif FLAGS.datasource == 'svhn':

			self.target_dataset = 'svhn'
			self.TRAIN_FILE = 'svhn_train.tfrecords'
			self.VALID_FILE = 'svhn_valid.tfrecords'
			self.TEST_FILE = 'svhn_test.tfrecords'

			self.input_dim = (32 * 32 * 3)
			self.input_height = 32
			self.input_width = 32
			self.input_channels = 3
			self.num_classes = 10
			self.dtype = tf.float32
			self.preprocess = self._preprocess_svhn
			self.get_dataset = self.get_tf_dataset

		elif FLAGS.datasource == 'cifar10':

			self.target_dataset = 'cifar10'
			self.TRAIN_FILE = 'train.tfrecords'
			self.VALID_FILE = 'validation.tfrecords'
			self.TEST_FILE = 'eval.tfrecords'

			self.input_dim = (32 * 32 * 3)
			self.input_height = 32
			self.input_width = 32
			self.input_channels = 3
			self.num_classes = 10
			self.dtype = tf.float32
			self.preprocess = self._preprocess_cifar10
			self.get_dataset = self.get_cifar10_tf_dataset

		# minor changes
		elif FLAGS.datasource == 'celebA':

			self.target_dataset = 'celebA'
			self.TRAIN_FILE = 'celebA_train.tfrecords'
			self.VALID_FILE = 'celebA_valid.tfrecords'
			self.TEST_FILE = 'celebA_test.tfrecords'

			# TODO: this is specific to the aligned+cropped (64 x 64) images
			self.input_dim = (64 * 64 * 3)
			self.input_height = 64
			self.input_width = 64
			self.input_channels = 3
			self.dtype = tf.float32
			self.preprocess = self._preprocess_celebA
			self.get_dataset = self.get_tf_dataset_celebA
		else:
			raise NotImplementedError

		train_dataset = self.get_dataset('train')

		return

	def _preprocess_omniglot(self, parsed_example):

		# otherwise this gets decoded weird...
		image = tf.decode_raw(parsed_example['features'], tf.float64)
		image = tf.cast(image, tf.float32)
		image.set_shape([self.input_dim])
		label = tf.cast(parsed_example['label'], tf.int32)
		# TODO: there are more labels here but i think it's fine for now

		return image, label

	def _preprocess_mnist(self, parsed_example):

		image = tf.decode_raw(parsed_example['features'], tf.uint8)
		image.set_shape([self.input_dim])
		image = tf.cast(image, tf.float32) * (1. / 255)
		label = tf.cast(parsed_example['label'], tf.int32)

		return image, label

	def _preprocess_binary_mnist(self, parsed_example):

		image = tf.decode_raw(parsed_example['features'], tf.float32)
		image.set_shape([self.input_dim])
		# labels not available for BinaryMNIST

		return image

	def _preprocess_svhn(self, parsed_example):

		image = tf.decode_raw(parsed_example['features'], tf.uint8)
		image = tf.reshape(image, (self.input_height, self.input_width, self.input_channels))
		image = tf.cast(image, tf.float32) * (1. / 255)
		label = tf.cast(parsed_example['label'], tf.int32)

		return image, label

	def _preprocess_cifar10(self, parsed_example):

		# the automatically generated cifar10 tfrecord has field "image" instead of "features"
		image = tf.decode_raw(parsed_example['image'], tf.uint8)
		image = tf.reshape(image, (self.input_channels, self.input_height, self.input_width))
		image = tf.transpose(image, [1, 2, 0])
		image = tf.cast(image, tf.float32) * (1. / 255)
		label = tf.cast(parsed_example['label'], tf.int32)

		return image, label

	def _preprocess_celebA(self, parsed_example):

		image = tf.decode_raw(parsed_example['features'], tf.uint8)
		# image = tf.decode_raw(parsed_example['image_raw'], tf.uint8)
		image = tf.reshape(image, (self.input_height, self.input_width, self.input_channels))
		# convert from bytes to data
		image = tf.divide(tf.to_float(image), 127.5) - 1.0
		# convert back to [0, 1] pixels
		image = tf.clip_by_value(tf.divide(image + 1., 2.), 0., 1.)

		return image

	def _test_celebA(self):
		record_iterator = tf.python_io.tf_record_iterator(path=os.path.join(FLAGS.datadir, self.target_dataset, self.TRAIN_FILE))

		for img_id, string_record in enumerate(record_iterator):
			
			example = tf.train.Example()
			example.ParseFromString(string_record)
			
			print(int(example.features.feature['height'].int64_list.value))
			height = int(example.features.feature['height']
										 .int64_list
										 .value[0])
			
			width = int(example.features.feature['width']
										.int64_list
										.value[0])

			depth = int(example.features.feature['depth']
										.int64_list
										.value[0])
			
			img_string = (example.features.feature['image_raw']
										  .bytes_list
										  .value[0])
			
			img_1d = np.fromstring(img_string, dtype=np.uint8)
			print(img_1d.shape)
			reconstructed_img = img_1d.reshape((height, width, -1))
			print(reconstructed_img.shape)
			import skimage.io as io
			io.imshow(reconstructed_img)
			io.show()
			if img_id == 5:
				exit()		

	def get_tf_dataset_celebA(self, split):

		def _parse_function(example_proto):
			example = {'features': tf.FixedLenFeature((), tf.string, default_value=''),
						# 'height': tf.FixedLenFeature((), tf.int64, default_value=218),
						# 'width': tf.FixedLenFeature((), tf.int64, default_value=178),
						# 'channels': tf.FixedLenFeature((), tf.int64, default_value=3)
						}
			parsed_example = tf.parse_single_example(example_proto, example)
			preprocessed_features = self.preprocess(parsed_example)
			return preprocessed_features

		filename = os.path.join(FLAGS.datadir, self.target_dataset, self.TRAIN_FILE if split=='train' 
			else self.VALID_FILE if split=='valid' else self.TEST_FILE)
		tf_dataset = tf.data.TFRecordDataset(filename)
		return tf_dataset.map(_parse_function)

	def get_cifar10_tf_dataset(self, split):

		def _parse_function(example_proto):
			example = {'image': tf.FixedLenFeature((), tf.string, default_value=''),
						  'label': tf.FixedLenFeature((), tf.int64, default_value=0)}
			parsed_example = tf.parse_single_example(example_proto, example)
			preprocessed_features, preprocessed_label = self.preprocess(parsed_example)
			return preprocessed_features, preprocessed_label

		filename = os.path.join(FLAGS.datadir, self.target_dataset, self.TRAIN_FILE if split=='train' 
			else self.VALID_FILE if split=='valid' else self.TEST_FILE)
		tf_dataset = tf.data.TFRecordDataset(filename)
		return tf_dataset.map(_parse_function)

	def get_tf_dataset(self, split):

		def _parse_function(example_proto):
			example = {'features': tf.FixedLenFeature((), tf.string, default_value=''),
						  'label': tf.FixedLenFeature((), tf.int64, default_value=0)}
			parsed_example = tf.parse_single_example(example_proto, example)
			preprocessed_features, preprocessed_label = self.preprocess(parsed_example)
			
			return preprocessed_features, preprocessed_label

		filename = os.path.join(FLAGS.datadir, self.target_dataset, self.TRAIN_FILE if split=='train' 
			else self.VALID_FILE if split=='valid' else self.TEST_FILE)
		tf_dataset = tf.data.TFRecordDataset(filename)
		return tf_dataset.map(_parse_function)

	def get_binary_tf_dataset(self, split):

		def _parse_function(example_proto):
			# no labels available for binary MNIST
			example = {'features': tf.FixedLenFeature((), tf.string, default_value='')}
			parsed_example = tf.parse_single_example(example_proto, example)
			preprocessed_features = self.preprocess(parsed_example)
			return preprocessed_features

		filename = os.path.join(FLAGS.datadir, self.target_dataset, self.TRAIN_FILE if split=='train' 
			else self.VALID_FILE if split=='valid' else self.TEST_FILE)
		tf_dataset = tf.data.TFRecordDataset(filename)
		return tf_dataset.map(_parse_function)