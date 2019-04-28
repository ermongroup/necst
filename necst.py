# NOTE: most of code from aditya-grover's UAE project (TODO: find link)
from utils import *
import tensorflow as tf 
import numpy as np
import time
import sys
from math import log, exp
from scipy.special import expit
from datasource import Datasource
from tensorflow.contrib.framework.python.framework import checkpoint_utils

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import itertools
import tensorflow.contrib.distributions as tfd
from noisy_channels import *
from tensorflow.contrib.distributions import Bernoulli, Categorical, RelaxedBernoulli, Normal
import pickle
from itertools import product, chain
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS


class NECST():
	def __init__(self, sess, datasource):

		self.seed = FLAGS.seed
		tf.set_random_seed(self.seed)
		np.random.seed(self.seed)

		self.sess = sess
		self.datasource = datasource
		self.input_dim = self.datasource.input_dim
		if self.input_dim == 784:
			self.img_dim = 28
		elif self.input_dim == 100:
			self.img_dim = 100
		elif self.input_dim == 7840 or self.input_dim == 3920:
			self.img_dim = 28
		elif self.input_dim == (32 * 32 * 3):
			self.img_dim = 32
		else:
			# celebA
			self.img_dim = 64
		self.z_dim = FLAGS.num_measurements
		self.dec_layers = [self.input_dim] + FLAGS.dec_arch
		self.enc_layers = FLAGS.enc_arch + [self.z_dim]
		self.transfer = FLAGS.transfer

		self.last_layer_act = tf.nn.sigmoid if FLAGS.non_linear_act else None

		# perturbation experiment
		self.noisy_mnist = FLAGS.noisy_mnist

		# for vimco
		self.is_binary = FLAGS.is_binary
		self.vimco_samples = FLAGS.vimco_samples

		# other params
		self.activation = FLAGS.activation
		self.lr = FLAGS.lr 
		# if need to use REINFORCE-like optimization scheme
		if not self.discrete_relax:
			self.theta_optimizer = FLAGS.optimizer(learning_rate=self.lr)
			self.phi_optimizer = FLAGS.optimizer(learning_rate=self.lr)
		else:
			# gumbel-softmax doesn't require 2 optimizers
			self.optimizer = FLAGS.optimizer
		self.training = True

		# noise levels
		self.channel_model = FLAGS.channel_model
		self.perturb_probs = FLAGS.perturb_probs
		self.test_perturb_probs = FLAGS.test_perturb_probs

		# mask
		self.mask = tf.placeholder_with_default(
			np.ones((FLAGS.batch_size, self.z_dim, 3)), shape=[None, self.z_dim, 3])
		# TODO: hacky - fix later
		if self.img_dim == 64:
			self.x = tf.placeholder(self.datasource.dtype, shape=[None, self.img_dim, self.img_dim, 3], name='vae_input')
		elif self.img_dim == 28:
			self.x = tf.placeholder(tf.float32, shape=[None, self.input_dim], name='vae_input')
		else:
			# svhn and cifar10
			self.x = tf.placeholder(tf.float32, shape=[None, self.img_dim, self.img_dim, 3], name='vae_input')

		# CS settings
		self.noise_std = tf.placeholder_with_default(FLAGS.noise_std, shape=(), name='noise_std')
		self.reg_param = tf.placeholder_with_default(FLAGS.reg_param, shape=(), name='reg_param')

		# gumbel-softmax and vimco-compatible; only discrete bits
		if self.img_dim == 64:
			self.mean, self.z, self.classif_z, self.q, self.x_reconstr_logits = self.celebA_create_collapsed_computation_graph(self.x, std=self.noise_std)
		else:
			# MNIST
			if self.channel_model == 'bsc':
				self.mean, self.z, self.classif_z, self.q, self.x_reconstr_logits = self.create_collapsed_computation_graph(self.x, std=self.noise_std)
			else:
				self.mean, self.z, self.q, self.x_reconstr_logits = self.create_erasure_collapsed_computation_graph(self.x, std=self.noise_std)
		if self.channel_model == 'bsc':
			self.test_mean, self.test_z, self.test_classif_z, self.test_q, self.test_x_reconstr_logits = self.get_collapsed_stochastic_test_sample(self.x, std=self.noise_std)
		else:
			self.test_mean, self.test_z, self.test_q, self.test_x_reconstr_logits = self.get_collapsed_erasure_stochastic_test_sample(self.x, std=self.noise_std)
		if not self.discrete_relax:
			print('using vimco loss...')
			if self.noisy_mnist:
				print('training with noisy MNIST, using true x values for vimco loss...')
				self.theta_loss, self.phi_loss, self.reconstr_loss = self.vimco_loss(
					self.true_x, self.x_reconstr_logits)
			else:
				self.theta_loss, self.phi_loss, self.reconstr_loss = self.vimco_loss(self.x, self.x_reconstr_logits)
		else:
			self.loss, self.reconstr_loss = self.get_loss(self.x, self.x_reconstr_logits)
		
		# loss calculation
		if self.noisy_mnist:
			print('training with noisy MNIST, using true x values for vimco loss...')
			self.test_loss = self.get_test_loss(self.true_x, self.test_x_reconstr_logits)
		else:
			self.test_loss = self.get_test_loss(self.x, self.test_x_reconstr_logits)

		# session ops
		self.global_step = tf.Variable(0, name='global_step', trainable=False)

		# set up optimization op
		if not self.discrete_relax:
			print('SETUP: using mutliple train ops due to discrete latent variable')
			# get decoder and encoder variables
			theta_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model/decoder')
			self.theta_vars = theta_vars
			self.theta_grads, variables = zip(*self.theta_optimizer.compute_gradients(self.theta_loss, var_list=theta_vars))
			self.discrete_train_op1 = self.theta_optimizer.minimize(self.theta_loss, global_step=self.global_step, var_list=theta_vars)

			# encoder variables
			phi_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model/encoder')
			self.phi_vars = phi_vars
			self.phi_grads, variables = zip(*self.phi_optimizer.compute_gradients(self.phi_loss, var_list=phi_vars))
			self.discrete_train_op2 = self.phi_optimizer.minimize(self.phi_loss, global_step=self.global_step, var_list=phi_vars)
		else:
			# gumbel-softmax
			self.train_op = self.optimizer(learning_rate=self.lr).minimize(self.loss, 
				global_step=self.global_step, var_list=train_vars)
		self.vars = train_vars

		# summary ops
		self.summary_op = tf.summary.merge_all()

		# session ops
		self.init_op = tf.global_variables_initializer()
		self.saver = tf.train.Saver(max_to_keep=None)


	def encoder(self, x, reuse=True):
		"""
		Specifies the parameters for the mean and variance of p(y|x)
		"""
		e = x
		enc_layers = self.enc_layers
		regularizer = tf.contrib.layers.l2_regularizer(scale=self.reg_param)
		with tf.variable_scope('model', reuse=reuse):
			with tf.variable_scope('encoder', reuse=reuse):
				for layer_idx, layer_dim in enumerate(enc_layers[:-1]):
					e = tf.layers.dense(e, layer_dim, activation=tf.nn.leaky_relu, kernel_regularizer=regularizer, reuse=reuse, name='fc-'+str(layer_idx))
				if self.channel_model == 'bsc':
					z_mean = tf.layers.dense(e, self.z_dim, activation=None, use_bias=False, kernel_regularizer=regularizer, reuse=reuse, name='fc-'+str(len(enc_layers)-1))
				else:
					# N x D x 2 for erasure channel!
					z_mean = tf.layers.dense(e, self.z_dim * 2, activation=None, use_bias=False, kernel_regularizer=regularizer, reuse=reuse, name='fc-'+str(len(enc_layers)-1))
					z_mean = tf.reshape(z_mean, (-1, self.z_dim, 2))
		return z_mean


	def categorical_encoder(self, x, reuse=True):
		"""
		Specifies the parameters for the mean and variance of p(y|x)
		"""
		e = x
		enc_layers = self.enc_layers
		regularizer = tf.contrib.layers.l2_regularizer(scale=self.reg_param)
		with tf.variable_scope('model', reuse=reuse):
			with tf.variable_scope('encoder', reuse=reuse):
				for layer_idx, layer_dim in enumerate(enc_layers[:-1]):
					e = tf.layers.dense(e, layer_dim, activation=tf.nn.elu, kernel_regularizer=regularizer, reuse=reuse, name='fc-'+str(layer_idx))
				# temporarily switch this to a categorical!
				z_mean = tf.layers.dense(e, self.z_dim, activation=None, use_bias=False, kernel_regularizer=regularizer, reuse=reuse, name='fc-'+str(len(enc_layers)-1))
		return z_mean


	def complex_encoder(self, x, reuse=True):
		"""
		more complex encoder architecture for images with more than 1 color channel
		""" 
		enc_layers = self.enc_layers
		regularizer = tf.contrib.layers.l2_regularizer(scale=self.reg_param)
		with tf.variable_scope('model', reuse=reuse):
			with tf.variable_scope('encoder', reuse=reuse):
				conv1 = tf.layers.conv2d(x, 32, 4, strides=(2,2), padding="SAME", activation=tf.nn.elu, kernel_regularizer=regularizer, reuse=reuse, name='conv1')
				conv2 = tf.layers.conv2d(conv1, 32, 4, strides=(2,2), padding="SAME", activation=tf.nn.elu, kernel_regularizer=regularizer, reuse=reuse, name='conv2')
				conv3 = tf.layers.conv2d(conv2, 64, 4, strides=(2,2), padding="SAME", activation=tf.nn.elu, kernel_regularizer=regularizer, reuse=reuse, name='conv3')
				conv4 = tf.layers.conv2d(conv3, 64, 4, strides=(2,2), padding="SAME", activation=tf.nn.elu, kernel_regularizer=regularizer, reuse=reuse, name='conv4')
				conv5 = tf.layers.conv2d(conv4, 256, 4, padding="VALID", activation=tf.nn.elu, kernel_regularizer=regularizer, reuse=reuse, name='conv5')
				flattened = tf.reshape(conv5, (-1, 256*1*1))
				z_mean = tf.layers.dense(flattened, enc_layers[-1], activation=None, use_bias=False, kernel_regularizer=regularizer, reuse=reuse, name='fc-final')
		return z_mean


	def convolutional_32_encoder(self, x, reuse=True):
		"""
		FINAL ARCHITECTURE!!
		more complex encoder architecture for images with more than 1 color channel
		""" 
		# TODO: may have to do PCA + fc MLP
		enc_layers = self.enc_layers
		regularizer = tf.contrib.layers.l2_regularizer(scale=self.reg_param)
		with tf.variable_scope('model', reuse=reuse):
			with tf.variable_scope('encoder', reuse=reuse):
				conv1 = tf.layers.conv2d(x, 128, 2, strides=(2,2), padding="VALID", activation=tf.nn.relu, kernel_regularizer=regularizer, reuse=reuse, name='conv1')
				conv1 = tf.layers.batch_normalization(conv1)
				print('conv1 shape: {}'.format(conv1.get_shape()))

				conv2 = tf.layers.conv2d(conv1, 256, 2, strides=(2,2), padding="VALID", activation=tf.nn.relu, kernel_regularizer=regularizer, reuse=reuse, name='conv2')
				conv2 = tf.layers.batch_normalization(conv2)
				print('conv2 shape: {}'.format(conv2.get_shape()))

				conv3 = tf.layers.conv2d(conv2, 512, 2, strides=(2,2), padding="VALID", activation=tf.nn.relu, kernel_regularizer=regularizer, reuse=reuse, name='conv3')
				conv3 = tf.layers.batch_normalization(conv3)
				print('conv3 shape: {}'.format(conv3.get_shape()))

				flattened = tf.contrib.layers.flatten(conv3)
				z_mean = tf.layers.dense(flattened, enc_layers[-1], activation=None, use_bias=False, kernel_regularizer=regularizer, reuse=reuse, name='fc-final')
		return z_mean


	def convolutional_32_decoder(self, z, reuse=True):
		"""
		FINAL ARCHITECTURE!!
		more complex decoder architecture for images with more than 1 color channel (e.g. celebA)
		"""
		z = tf.convert_to_tensor(z)
		reuse=tf.AUTO_REUSE

		if self.vimco_samples > 1:
			samples = []

		with tf.variable_scope('model', reuse=reuse):
			with tf.variable_scope('decoder', reuse=reuse):
				if len(z.get_shape().as_list()) == 2:
					# test
					d = tf.layers.dense(z, 4*4*512, activation=tf.nn.relu, use_bias=False, reuse=reuse, name='fc1')	
					# print('init d shape: {}'.format(d.get_shape()))	
					d = tf.reshape(d, (-1, 4, 4, 512))
					# print('d shape: {}'.format(d.get_shape()))
					deconv1 = tf.layers.conv2d_transpose(d, 512, 2, strides=(2,2), padding="VALID", activation=tf.nn.relu, reuse=reuse, name='deconv1')
					deconv1 = tf.layers.batch_normalization(deconv1)

					# print('deconv1 shape: {}'.format(deconv1.get_shape()))
					deconv2 = tf.layers.conv2d_transpose(deconv1, 256, 2, strides=(2,2), padding="VALID", activation=tf.nn.relu, reuse=reuse, name='deconv2')
					deconv2 = tf.layers.batch_normalization(deconv2)

					# print('deconv2 shape: {}'.format(deconv2.get_shape()))
					deconv3 = tf.layers.conv2d_transpose(deconv2, 128, 2, strides=(2,2), padding="VALID", activation=tf.nn.relu, reuse=reuse, name='deconv3')
					deconv3 = tf.layers.batch_normalization(deconv3)
					print('deconv3 shape: {}'.format(deconv3.get_shape()))

					deconv4 = tf.layers.conv2d(deconv3, 3, 1, strides=(1,1), padding="VALID", activation=self.last_layer_act, reuse=reuse, name='deconv4')
					return deconv4
				else:
					# train
					for i in range(self.vimco_samples):
						# iterate through one vimco sample at a time
						z_sample = z[i]
						d = tf.layers.dense(z_sample, 4*4*512, activation=tf.nn.relu, use_bias=False, reuse=reuse, name='fc1')	
						d = tf.reshape(d, (-1, 4, 4, 512))
						# print('d shape: {}'.format(d.get_shape()))
						deconv1 = tf.layers.conv2d_transpose(d, 512, 2, strides=(2,2), padding="VALID", activation=tf.nn.relu, reuse=reuse, name='deconv1')
						deconv1 = tf.layers.batch_normalization(deconv1)

						# print('deconv1 shape: {}'.format(deconv1.get_shape()))
						deconv2 = tf.layers.conv2d_transpose(deconv1, 256, 2, strides=(2,2), padding="VALID", activation=tf.nn.relu, reuse=reuse, name='deconv2')
						deconv2 = tf.layers.batch_normalization(deconv2)

						# print('deconv2 shape: {}'.format(deconv2.get_shape()))
						deconv3 = tf.layers.conv2d_transpose(deconv2, 128, 2, strides=(2,2), padding="VALID", activation=tf.nn.relu, reuse=reuse, name='deconv3')
						deconv3 = tf.layers.batch_normalization(deconv3)
						print('deconv3 shape: {}'.format(deconv3.get_shape()))
						
						# print('deconv3 shape: {}'.format(deconv3.get_shape()))
						deconv4 = tf.layers.conv2d(deconv3, 3, 1, strides=(1,1), padding="VALID", activation=tf.nn.sigmoid, reuse=reuse, name='deconv4')
						samples.append(deconv4)
		x_reconstr_logits = tf.stack(samples, axis=0)
		print(x_reconstr_logits.get_shape())
		return x_reconstr_logits	


	def cifar10_convolutional_encoder(self, x, reuse=True):
		"""
		more complex encoder architecture for images with more than 1 color channel
		--> architecture specifically for cifar10!
		""" 
		# print('x shape: {}'.format(x.get_shape()))
		enc_layers = self.enc_layers
		regularizer = tf.contrib.layers.l2_regularizer(scale=self.reg_param)
		with tf.variable_scope('model', reuse=reuse):
			with tf.variable_scope('encoder', reuse=reuse):
				conv1 = tf.layers.conv2d(x, 64, (3,3), padding="SAME", activation=None, kernel_regularizer=regularizer, reuse=reuse, name='conv1')
				# print('conv1 shape: {}'.format(conv1.get_shape()))
				bn1 = tf.layers.batch_normalization(conv1)
				relu1 = tf.nn.relu(bn1)
				conv1_out = tf.layers.max_pooling2d(relu1, (2,2), (2,2), padding='same')
				# 2nd convolutional layer
				conv2 = tf.layers.conv2d(conv1_out, 32, (3,3), padding="SAME", activation=None, kernel_regularizer=regularizer, reuse=reuse, name='conv2')
				# print('conv2 shape: {}'.format(conv2.get_shape()))
				bn2 = tf.layers.batch_normalization(conv2)
				relu2 = tf.nn.relu(bn2)
				conv2_out = tf.layers.max_pooling2d(relu2, (2,2), (2,2), padding='same')
				# 3rd convolutional layer
				conv3 = tf.layers.conv2d(conv2_out, 16, (3,3), padding="SAME", activation=None, kernel_regularizer=regularizer, reuse=reuse, name='conv3')
				bn3 = tf.layers.batch_normalization(conv3)
				relu3 = tf.nn.relu(bn3)
				conv3_out = tf.layers.max_pooling2d(relu3, (2,2), (2,2), padding='same')
				# print('after 3 convolutions, shape: {}'.format(conv3_out.get_shape()))
				flattened = tf.reshape(conv3_out, (-1, 4*4*16))
				z_mean = tf.layers.dense(flattened, enc_layers[-1], activation=None, use_bias=False, kernel_regularizer=regularizer, reuse=reuse, name='fc-final')
		return z_mean


	def cifar10_convolutional_decoder(self, z, reuse=True):
		"""
		more complex decoder architecture for images with more than 1 color channel
		--> NOTE: this architecture is specifically tailored for CIFAR10!
		"""
		z = tf.convert_to_tensor(z)
		reuse=tf.AUTO_REUSE

		if self.vimco_samples > 1:
			samples = []

		with tf.variable_scope('model', reuse=reuse):
			with tf.variable_scope('decoder', reuse=reuse):
				if len(z.get_shape().as_list()) == 2:
					# reshape input properly for deconvolution
					d = tf.layers.dense(z, 4*4*16, activation=None, use_bias=False, reuse=reuse, name='fc1')	
					d = tf.reshape(d, (-1, 4, 4, 16))
					# start deconvolution process
					deconv1 = tf.layers.conv2d(d, 16, (3,3), padding="SAME", activation=None, 
						reuse=reuse, name='deconv1')
					# print('deconv1 shape: {}'.format(deconv1.get_shape()))
					bn1 = tf.layers.batch_normalization(deconv1)
					relu1 = tf.nn.relu(bn1)
					deconv1_out = tf.keras.layers.UpSampling2D((2,2))(relu1)
					# 2nd deconvolutional layer
					deconv2 = tf.layers.conv2d(deconv1_out, 32, (3,3), padding="SAME", activation=None, 
						reuse=reuse, name='deconv2')
					# print('deconv2 shape: {}'.format(deconv2.get_shape()))
					bn2 = tf.layers.batch_normalization(deconv2)
					relu2 = tf.nn.relu(bn2)
					deconv2_out = tf.keras.layers.UpSampling2D((2,2))(relu2)
					# 3rd convolutional layer
					deconv3 = tf.layers.conv2d(deconv2_out, 64, (3,3), padding="SAME", activation=None, 
						reuse=reuse, name='deconv3')
					# print('deconv3 shape: {}'.format(deconv3.get_shape()))
					bn3 = tf.layers.batch_normalization(deconv3)
					relu3 = tf.nn.relu(bn3)
					out = tf.keras.layers.UpSampling2D((2,2))(relu3)
					deconv3_out = tf.layers.conv2d(out, 3, (3, 3), padding="SAME", activation=None)
					deconv3_out = tf.layers.batch_normalization(deconv3_out)

					deconv3_out = tf.nn.sigmoid(deconv3_out)
					# print('deconv3 shape: {}'.format(deconv3_out.get_shape()))
					return deconv3_out
				else:
					# train
					for i in range(self.vimco_samples):
						# print('in vimco samples part of deconvolution!')
						# iterate through one vimco sample at a time
						z_sample = z[i]
						# print('z_sample shape: {}'.format(z_sample.get_shape()))
						# reshape input properly for deconvolution
						d = tf.layers.dense(z_sample, 4*4*16, activation=None, use_bias=False, reuse=reuse, name='fc1')	
						d = tf.reshape(d, (-1, 4, 4, 16))
						# start deconvolution process
						deconv1 = tf.layers.conv2d(d, 16, (3,3), padding="SAME", activation=None, reuse=reuse, name='deconv1')
						# print('deconv1 shape: {}'.format(deconv1.get_shape()))
						bn1 = tf.layers.batch_normalization(deconv1)
						relu1 = tf.nn.relu(bn1)
						deconv1_out = tf.keras.layers.UpSampling2D((2,2))(relu1)
						# 2nd deconvolutional layer
						deconv2 = tf.layers.conv2d(deconv1_out, 32, (3,3), padding="SAME", activation=None, 
							reuse=reuse, name='deconv2')
						# print('deconv2 shape: {}'.format(deconv2.get_shape()))
						bn2 = tf.layers.batch_normalization(deconv2)
						relu2 = tf.nn.relu(bn2)
						deconv2_out = tf.keras.layers.UpSampling2D((2,2))(relu2)
						# 3rd convolutional layer
						deconv3 = tf.layers.conv2d(deconv2_out, 64, (3,3), padding="SAME", activation=None, 
							reuse=reuse, name='deconv3')
						# print('deconv3 shape: {}'.format(deconv3.get_shape()))
						bn3 = tf.layers.batch_normalization(deconv3)
						relu3 = tf.nn.relu(bn3)
						out = tf.keras.layers.UpSampling2D((2,2))(relu3)
						# print('out shape: {}'.format(out.get_shape()))
						deconv3_out = tf.layers.conv2d(out, 3, (3, 3), padding="SAME", activation=None)
						deconv3_out = tf.layers.batch_normalization(deconv3_out)
						deconv3_out = tf.nn.sigmoid(deconv3_out)
						# print('deconv3_out shape: {}'.format(deconv3_out.get_shape()))
						samples.append(deconv3_out)
		x_reconstr_logits = tf.stack(samples, axis=0)
		print(x_reconstr_logits.get_shape())
		return x_reconstr_logits


	def convolutional_decoder(self, z, reuse=True):
		"""
		more complex decoder architecture for images with more than 1 color channel (e.g. celebA)
		"""
		z = tf.convert_to_tensor(z)
		reuse=tf.AUTO_REUSE

		if self.vimco_samples > 1:
			samples = []

		with tf.variable_scope('model', reuse=reuse):
			with tf.variable_scope('decoder', reuse=reuse):
				if len(z.get_shape().as_list()) == 2:
					# test
					d = tf.layers.dense(z, 4*4*32, activation=tf.nn.elu, use_bias=False, reuse=reuse, name='fc1')	
					print('init d shape: {}'.format(d.get_shape()))	
					d = tf.reshape(d, (-1, 4, 4, 32))
					print('d shape: {}'.format(d.get_shape()))
					deconv1 = tf.layers.conv2d_transpose(d, 32, 1, strides=(2,2), padding="SAME", activation=tf.nn.elu, reuse=reuse, name='deconv1')
					print('deconv1 shape: {}'.format(deconv1.get_shape()))
					deconv2 = tf.layers.conv2d_transpose(deconv1, 32, 5, strides=(2,2), padding="SAME", activation=tf.nn.elu, reuse=reuse, name='deconv2')
					print('deconv2 shape: {}'.format(deconv2.get_shape()))
					deconv3 = tf.layers.conv2d_transpose(deconv2, 1, 5, strides=(2,2), padding="SAME", activation=tf.nn.sigmoid, reuse=reuse, name='deconv3')
					print('deconv3 shape: {}'.format(deconv3.get_shape()))
					return deconv3
				else:
					# train
					for i in range(self.vimco_samples):
						# iterate through one vimco sample at a time
						z_sample = z[i]
						d = tf.layers.dense(z, 4*4*32, activation=tf.nn.elu, use_bias=False, reuse=reuse, name='fc1')	
						d = tf.reshape(d, (-1, 4, 4, 32))
						print('d shape: {}'.format(d.get_shape()))
						deconv1 = tf.layers.conv2d_transpose(d, 32, 1, strides=(2,2), padding="SAME", activation=tf.nn.elu, reuse=reuse, name='deconv1')
						print('deconv1 shape: {}'.format(deconv1.get_shape()))
						deconv2 = tf.layers.conv2d_transpose(deconv1, 32, 5, strides=(2,2), padding="SAME", activation=tf.nn.elu, reuse=reuse, name='deconv2')
						print('deconv2 shape: {}'.format(deconv2.get_shape()))
						deconv3 = tf.layers.conv2d_transpose(deconv2, 1, 5, strides=(2,2), padding="SAME", activation=tf.nn.sigmoid, reuse=reuse, name='deconv3')
						print('deconv3 shape: {}'.format(deconv3.get_shape()))
						samples.append(deconv3)
		x_reconstr_logits = tf.stack(samples, axis=0)
		print(x_reconstr_logits.get_shape())
		return x_reconstr_logits


	def complex_decoder(self, z, reuse=True):
		"""
		more complex decoder architecture for images with more than 1 color channel (e.g. celebA)
		"""
		z = tf.convert_to_tensor(z)
		reuse=tf.AUTO_REUSE

		if self.vimco_samples > 1:
			samples = []

		with tf.variable_scope('model', reuse=reuse):
			with tf.variable_scope('decoder', reuse=reuse):
				if len(z.get_shape().as_list()) == 2:
					# test
					d = tf.layers.dense(z, 256, activation=tf.nn.elu, use_bias=False, reuse=reuse, name='fc1')		
					d = tf.reshape(d, (-1, 1, 1, 256))
					deconv1 = tf.layers.conv2d_transpose(d, 256, 4, padding="VALID", activation=tf.nn.elu, reuse=reuse, name='deconv1')
					deconv2 = tf.layers.conv2d_transpose(deconv1, 64, 4, strides=(2,2), padding="SAME", activation=tf.nn.elu, reuse=reuse, name='deconv2')
					deconv3 = tf.layers.conv2d_transpose(deconv2, 64, 4, strides=(2,2), padding="SAME", activation=tf.nn.elu, reuse=reuse, name='deconv3')
					deconv4 = tf.layers.conv2d_transpose(deconv3, 32, 4, strides=(2,2), padding="SAME", activation=tf.nn.elu, reuse=reuse, name='deconv4')
					# output channel = 3
					deconv5 = tf.layers.conv2d_transpose(deconv4, 3, 4, strides=(2,2), padding="SAME", activation=tf.nn.sigmoid, reuse=reuse, name='deconv5')
					return deconv5
				else:
					# train
					for i in range(self.vimco_samples):
						# iterate through one vimco sample at a time
						z_sample = z[i]
						d = tf.layers.dense(z_sample, 256, activation=tf.nn.elu, use_bias=False, reuse=reuse, name='fc1')		
						d = tf.reshape(d, (-1, 1, 1, 256))
						deconv1 = tf.layers.conv2d_transpose(d, 256, 4, padding="VALID", activation=tf.nn.elu, reuse=reuse, name='deconv1')
						deconv2 = tf.layers.conv2d_transpose(deconv1, 64, 4, strides=(2,2), padding="SAME", activation=tf.nn.elu, reuse=reuse, name='deconv2')
						deconv3 = tf.layers.conv2d_transpose(deconv2, 64, 4, strides=(2,2), padding="SAME", activation=tf.nn.elu, reuse=reuse, name='deconv3')
						deconv4 = tf.layers.conv2d_transpose(deconv3, 32, 4, strides=(2,2), padding="SAME", activation=tf.nn.elu, reuse=reuse, name='deconv4')
						# output channel = 3
						deconv5 = tf.layers.conv2d_transpose(deconv4, 3, 4, strides=(2,2), padding="SAME", activation=tf.nn.sigmoid, reuse=reuse, name='deconv5')
						samples.append(deconv5)
		x_reconstr_logits = tf.stack(samples, axis=0)
		print(x_reconstr_logits.get_shape())
		return x_reconstr_logits


	def decoder(self, z, reuse=True, use_bias=False):
		# revert to original decoder for now!!

		d = tf.convert_to_tensor(z)
		dec_layers = self.dec_layers

		with tf.variable_scope('model', reuse=reuse):
			with tf.variable_scope('decoder', reuse=reuse):
				for layer_idx, layer_dim in list(reversed(list(enumerate(dec_layers))))[:-1]:
					d = tf.layers.dense(d, layer_dim, activation=tf.nn.leaky_relu, reuse=reuse, name='fc-' + str(layer_idx), use_bias=use_bias)
				# assume gaussian decoder
				x_reconstr_logits = tf.layers.dense(d, dec_layers[0], activation=self.last_layer_act, reuse=reuse, name='fc-0', use_bias=use_bias) # clip values between 0 and 1

		return x_reconstr_logits


	def get_loss(self, x, x_reconstr_logits):

		reg_loss = tf.losses.get_regularization_loss()
		if self.img_dim == 64:
			reconstr_loss = tf.reduce_mean(
				tf.reduce_sum(tf.squared_difference(x, x_reconstr_logits), axis=[1,2,3]))
		else:
			reconstr_loss = tf.reduce_mean(
				tf.reduce_sum(tf.squared_difference(x, x_reconstr_logits), axis=1))
			# TODO: binary MNIST
			# reconstr_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
							# logits=x_reconstr_logits, labels=x))
		tf.summary.scalar('reconstruction loss', reconstr_loss)
		total_loss = reconstr_loss + reg_loss
		
		return total_loss, reconstr_loss


	def build_vimco_loss(self, l):
	    """Builds VIMCO baseline as in https://arxiv.org/abs/1602.06725
	    Args:
	    l: Per-sample learning signal. shape [k, b] or
	        [number of samples, batch_size]
	    log_q_h: Sum of log q(h^l) over layers
	    Returns:
	    baseline to subtract from l
	    ---> jaan's implementation
	    """
	    # compute the multi-sample stochastic bound
	    k, b = l.get_shape().as_list()
	    if b is None:
	    	b = FLAGS.batch_size
	    kf = tf.cast(k, tf.float32)

	    l_logsumexp = tf.reduce_logsumexp(l, [0], keepdims=True)
	    # L_hat is the multi-sample stochastic bound
	    L_hat = l_logsumexp - tf.log(kf)
	    
	    # precompute the sum of log f
	    s = tf.reduce_sum(l, 0, keepdims=True)
	    
	    # compute baseline for each sample
	    diag_mask = tf.expand_dims(tf.diag(tf.ones([k], dtype=tf.float32)), -1)
	    off_diag_mask = 1. - diag_mask

	    diff = tf.expand_dims(s - l, 0)  # expand for proper broadcasting
	    l_i_diag = 1. / (kf - 1.) * diff * diag_mask
	    l_i_off_diag = off_diag_mask * tf.stack([l] * k)
	    l_i = l_i_diag + l_i_off_diag
	    L_hat_minus_i = tf.reduce_logsumexp(l_i, [1]) - tf.log(kf)
	    
	    # compute the importance weights
	    w = tf.stop_gradient(tf.exp((l - l_logsumexp)))
	    
	    # compute gradient contributions
	    local_l = tf.stop_gradient(L_hat - L_hat_minus_i)
	    
	    return local_l, w, L_hat[0, :]


	def vimco_loss(self, x, x_reconstr_logits):
		
		reg_loss = tf.losses.get_regularization_loss()
		if self.is_binary:
			print('using Gaussian decoder and rounding to get hard {0, 1} values')
		if self.img_dim == 64:
			reconstr_loss = tf.reduce_sum(tf.squared_difference(x, x_reconstr_logits), axis=[2,3,4])
		elif self.img_dim == 32 and self.datasource.target_dataset == 'cifar10':
			# reconstr_loss = tf.reduce_sum(
			# 	tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=x_reconstr_logits[0]))
			# TODO: ask if this should be gaussian or sigmoid cross entropy!!!!
			reconstr_loss = tf.reduce_sum(tf.squared_difference(x, x_reconstr_logits), axis=[2,3,4])
		elif self.img_dim == 32 and self.datasource.target_dataset == 'svhn':
			reconstr_loss = tf.reduce_sum(tf.squared_difference(x, x_reconstr_logits), axis=[2,3,4])
		else:
			reconstr_loss = tf.reduce_sum(tf.squared_difference(x, x_reconstr_logits), axis=-1)
		print('reconstr loss shape: {}'.format(reconstr_loss.get_shape()))

		# define your distribution q as a bernoulli, get multiple samples for VIMCO
		log_q_h_list = self.q.log_prob(self.z)
		log_q_h = tf.reduce_sum(log_q_h_list, axis=-1)
		
		# to be able to look at the log probabilities 
		self.log_q_h = log_q_h
		self.log_q_h_list = log_q_h_list
		loss = reconstr_loss

		# get vimco loss
		local_l, w, full_loss = self.build_vimco_loss(loss)

		# get appropriate losses for theta and phi respectively
		self.local_l = local_l
		# TODO: check if this should be elbo or reconstr_loss
		theta_loss = (w * reconstr_loss) # shapes are both (5, batch_size)
		phi_loss = (local_l * log_q_h) + theta_loss

		# first sum over each sample, then average over minibatch
		theta_loss = tf.reduce_mean(tf.reduce_sum(theta_loss, axis=0))
		phi_loss = tf.reduce_mean(tf.reduce_sum(phi_loss, axis=0)) + reg_loss
		# phi_loss = tf.reduce_mean(tf.reduce_sum(phi_loss, axis=0))
		full_loss = tf.reduce_mean(full_loss)

		# total_loss = reg_loss + full_loss
		tf.summary.scalar('vimco (no gradient reduction) loss', full_loss)

		# TODO
		return theta_loss, phi_loss, full_loss


	def get_test_loss(self, x, x_reconstr_logits):

		print('in function get_test_loss()')
		# reconstruction loss only, no regularization 
		if self.is_binary:
			print('self.is_binary is True; using Gaussian decoder')
		if self.img_dim == 64 or self.img_dim == 32:
			reconstr_loss = tf.reduce_mean(
				tf.reduce_sum(tf.squared_difference(x, x_reconstr_logits), axis=[1,2,3]))
		else:
			reconstr_loss = tf.reduce_mean(
				tf.reduce_sum(tf.squared_difference(x, x_reconstr_logits), axis=1))

		# return tf.reduce_mean(loss)
		return reconstr_loss


	# TODO: stefano's suggestion
	def create_collapsed_computation_graph(self, x, std=0.1, reuse=False):
		"""
		this models both (Y_i|X) and N as Bernoullis,
		so you get Y_i|X ~ Bern(sigmoid(WX) - 2*sigmoid(WX)*p + p)
		"""
		print('in create_collapsed_computation_graph()')
		print('TRAIN: implicitly flipping individual bits with probability {}'.format(self.perturb_probs))
		dset_name = self.datasource.target_dataset
		if dset_name == 'mnist':
			mean = self.encoder(x, reuse=reuse)
		elif dset_name == 'omniglot':
			# mean = self.convolutional_32_encoder(x, reuse=reuse)
			mean = self.encoder(x, reuse=reuse)
		elif dset_name == 'cifar10':
			mean = self.convolutional_32_encoder(x, reuse=reuse)
		elif dset_name == 'celebA':
			mean = self.complex_encoder(x, reuse=reuse)
		elif dset_name == 'svhn':
			mean = self.convolutional_32_encoder(x, reuse=reuse)
		else:
			print('my dataset name is: {} and im lazy'.format(dset_name))
			raise NotImplementedError

		# for downstream classification
		classif_q = Bernoulli(logits=mean)
		classif_y = tf.cast(classif_q.sample(), tf.float32)
		
		# if self.perturb_probs == 0, then you have to feed in logits for the Bernoulli to avoid NaNs
		if self.perturb_probs != 0:
			y_hat_prob = tf.nn.sigmoid(mean)
			if self.n_sigbits > 0:
				print('setting first {} bits to be noise-free!'.format(self.n_sigbits))
				adjusted = np.ones((FLAGS.batch_size, self.z_dim)) * self.perturb_probs
				adjusted[:, 0:self.n_sigbits] = 1e-6
				adjusted_tensor = tf.constant(adjusted, dtype=tf.float32)
				total_prob = y_hat_prob - (2 * y_hat_prob * adjusted_tensor) + adjusted_tensor
			else:
				total_prob = y_hat_prob - (2 * y_hat_prob * self.perturb_probs) + self.perturb_probs
			q = Bernoulli(probs=total_prob)
		else:
			print('no additional channel noise; feeding in logits for latent q_phi(z|x) to avoid numerical issues')
			total_prob = tf.nn.sigmoid(mean)
			q = Bernoulli(logits=mean)	

		# use VIMCO if self.vimco_samples > 1, else just one sample
		y = tf.cast(q.sample(self.vimco_samples), tf.float32)
		if dset_name == 'mnist':
			x_reconstr_logits = self.decoder(y, reuse=reuse)
		elif dset_name == 'omniglot':
			# x_reconstr_logits = self.convolutional_32_decoder(y, reuse=reuse)
			x_reconstr_logits = self.decoder(y, reuse=reuse)
		elif dset_name == 'cifar10':
			x_reconstr_logits = self.convolutional_32_decoder(y, reuse=reuse)
		elif dset_name == 'celebA':
			x_reconstr_logits = self.complex_decoder(y, reuse=reuse)
		elif dset_name == 'svhn':
			x_reconstr_logits = self.convolutional_32_decoder(y, reuse=reuse)
		else:
			print('my dataset name is: {} and im lazy'.format(dset_name))
			raise NotImplementedError

		# return mean, y, total_prob, q, x_reconstr_logits
		return total_prob, y, classif_y, q, x_reconstr_logits


	def create_erasure_collapsed_computation_graph(self, x, std=0.1, reuse=False):
		"""
		this models both (Y_i|X) and N as Bernoullis,
		so you get Y_i|X ~ Bern(sigmoid(WX) - 2*sigmoid(WX)*p + p)
		"""
		print('in create_collapsed_computation_graph() for ERASURE channel!!')
		print('TRAIN: implicitly flipping individual bits with probability {}'.format(self.perturb_probs))
		dset_name = self.datasource.target_dataset
		if dset_name == 'mnist' or dset_name == 'BinaryMNIST':
			mean = self.encoder(x, reuse=reuse)
		elif dset_name == 'omniglot':
			mean = self.convolutional_32_encoder(x, reuse=reuse)
		elif dset_name == 'cifar10':
			mean = self.cifar10_convolutional_encoder(x, reuse=reuse)
		else:
			print('my dataset name is: {} and im lazy'.format(dset_name))
			raise NotImplementedError

		# TODO!!!!!
		# for downstream classification
		# classif_q = Bernoulli(logits=mean)
		# classif_y = tf.cast(classif_q.sample(), tf.float32)
		
		# if self.perturb_probs == 0, then you have to feed in logits for the Bernoulli to avoid NaNs
		if self.perturb_probs != 0:
			print('computing probabilities for erasure channel!')
			# TODO
			y_hat_prob = tf.nn.softmax(mean)
			y_hat_prob = tf.clip_by_value(y_hat_prob, 1e-7, 1.-1e-7)

			# construct mask for erasure channel
			mask = np.zeros((2,3))
			mask[0,0] = 1 - self.perturb_probs
			mask[0,2] = self.perturb_probs
			mask[1,1] = 1 - self.perturb_probs
			mask[1,2] = self.perturb_probs

			total_prob = tf.reshape(tf.reshape(y_hat_prob, [-1, 2]) @ mask, [-1, self.z_dim, 3])
			total_prob = tf.clip_by_value(total_prob, 1e-7, 1.-1e-7)
			q = Categorical(probs=total_prob)
		else:
			print('use BSC channel if you want to run for noise=0!')
			raise NotImplementedError	

		# use VIMCO if self.vimco_samples > 1, else just one sample
		y = tf.cast(q.sample(self.vimco_samples), tf.float32)

		print('y shape: {}'.format(y.get_shape()))

		if dset_name == 'mnist' or dset_name == 'BinaryMNIST':
			x_reconstr_logits = self.decoder(y, reuse=reuse)
		elif dset_name == 'omniglot':
			x_reconstr_logits = self.convolutional_32_decoder(y, reuse=reuse)
		elif dset_name == 'cifar10':
			x_reconstr_logits = self.cifar10_convolutional_decoder(y, reuse=reuse)
		else:
			print('my dataset name is: {} and im lazy'.format(dset_name))
			raise NotImplementedError

		return mean, y, q, x_reconstr_logits
		# return total_prob, y, classif_y, q, x_reconstr_logits


	# TODO: vanilla beta-VAE for celebA
	def celebA_create_collapsed_computation_graph(self, x, std=0.1, reuse=False):
		"""
		this models both (Y_i|X) and N as Bernoullis,
		so you get Y_i|X ~ Bern(sigmoid(WX) - 2*sigmoid(WX)*p + p)
		"""
		print('TRAIN: implicitly flipping individual bits with probability {}'.format(self.perturb_probs))
		mean = self.complex_encoder(x, reuse=reuse)

		# classif_y
		classif_y = tf.cast(Bernoulli(logits=mean).sample(), tf.float32)
		
		# if self.perturb_probs == 0, then you have to feed in logits for the Bernoulli to avoid NaNs
		if self.perturb_probs != 0:
			y_hat_prob = tf.nn.sigmoid(mean)
			if self.n_sigbits > 0:
				print('setting first {} bits to be noise-free!'.format(self.n_sigbits))
				adjusted = np.ones((FLAGS.batch_size, self.z_dim)) * self.perturb_probs
				adjusted[:, 0:self.n_sigbits] = 1e-6
				adjusted_tensor = tf.constant(adjusted, dtype=tf.float32)
				total_prob = y_hat_prob - (2 * y_hat_prob * adjusted_tensor) + adjusted_tensor
			else:
				total_prob = y_hat_prob - (2 * y_hat_prob * self.perturb_probs) + self.perturb_probs
			q = Bernoulli(probs=total_prob)
		else:
			print('no additional channel noise; feeding in logits for latent q_phi(z|x) to avoid numerical issues')
			q = Bernoulli(logits=mean)

		# use VIMCO if self.vimco_samples > 1, else just one sample
		y = tf.cast(q.sample(self.vimco_samples), tf.float32)
		# y = tf.cast(q.sample(), tf.float32)
		print('y shape: {}'.format(y.get_shape()))

		x_reconstr_logits = self.complex_decoder(y, reuse=reuse)
		print('x reconstr logits shape: {}'.format(x_reconstr_logits.get_shape()))

		return mean, y, classif_y, q, x_reconstr_logits


	def create_stochastic_computation_graph(self, x, std=0.1, reuse=False):
		"""
		computation graph designed to work with gumbel-softmax
		"""
		mean = self.encoder(x, reuse=reuse)

		if self.discrete_relax:
			# gumbel-softmax
			raise NotImplementedError
		else:
			print('creating stochastic computation graph')
			# TODO: this Bernoulli should be deterministic!!!
			q = Bernoulli(logits=mean)
			# VIMCO 
			if self.vimco_samples > 1:
				print('using VIMCO')
				y_hat = tf.cast(q.sample(self.vimco_samples), tf.float32)
				if self.perturb_probs != 0.:
					print('TRAIN: flipping individual bits with probability {}'.format(self.perturb_probs))
					# error drawn from independent Bernoullis
					err_prob = tf.ones_like(mean) * self.perturb_probs
					# TODO: will you have to save self.N???
					N = tf.cast(Bernoulli(probs=err_prob).sample(self.vimco_samples), tf.float32)

					# add in the channel noise
					mod_op = tf.ones_like(y_hat) * 2.
					y = tf.mod(y_hat + N, mod_op)
				else:
					# if no channel noise, just feed in y_hat
					y = y_hat
			else:
				print('sampling one discrete value')
				# TODO: this Bernoulli should be deterministic!!!
				q = Bernoulli(logits=mean)
				y_hat = tf.cast(q.sample(), tf.float32)
				if self.perturb_probs != 0.:
					print('TRAIN: flipping individual bits with probability {}'.format(self.perturb_probs))
					err_prob = tf.ones_like(mean) * self.perturb_probs
					# TODO: will you have to save self.N???
					N = tf.cast(Bernoulli(probs=err_prob).sample(), tf.float32)
					# add in the channel noise
					mod_op = tf.ones_like(y_hat) * 2.
					y = tf.mod(y_hat + N, mod_op)
				else:
					# if no channel noise, just feed in y_hat
					y = y_hat

		x_reconstr_logits = self.decoder(y, reuse=reuse)

		return mean, y, q, x_reconstr_logits


	def get_test_sample(self, x, std=0.1, reuse=False):

		soft_bits = self.deterministic_encoder(x, reuse=tf.AUTO_REUSE)
		if not self.sigmoid:
			z = tf.round(soft_bits)
		else:
			# really high number for now
			z = tf.nn.sigmoid(100. * soft_bits)

		# TODO: channel noise
		if self.perturb_probs != 0.:
			print('TEST: flipping individual bits with probability {}'.format(self.perturb_probs))
			err_prob = tf.ones_like(soft_bits) * self.perturb_probs
			N = tf.cast(Bernoulli(probs=err_prob).sample(), tf.float32)
			mod_op = tf.ones_like(soft_bits) * 2.
			z = tf.mod(z + N, mod_op)
		x_reconstr_logits = self.decoder(z, reuse=tf.AUTO_REUSE)

		return soft_bits, z, x_reconstr_logits


	# TODO: stefano's suggestion
	def get_collapsed_stochastic_test_sample(self, x, std=0.1, reuse=False):
		"""
		use collapsed Bernoulli at test time as well
		"""
		print('TEST: implicitly flipping individual bits with probability {}'.format(self.test_perturb_probs))
		dset_name = self.datasource.target_dataset
		if dset_name == 'mnist':
			mean = self.encoder(x, reuse=tf.AUTO_REUSE)
		elif dset_name == 'omniglot':
			# mean = self.convolutional_32_encoder(x, reuse=tf.AUTO_REUSE)
			mean = self.encoder(x, reuse=tf.AUTO_REUSE)
		elif dset_name == 'cifar10':
			mean = self.convolutional_32_encoder(x, reuse=tf.AUTO_REUSE)
		elif dset_name == 'svhn':
			mean = self.convolutional_32_encoder(x, reuse=tf.AUTO_REUSE)
		elif dset_name == 'celebA':
			mean = self.complex_encoder(x, reuse=tf.AUTO_REUSE)
		else:
			print('my dataset name is: {} and im lazy'.format(dset_name))
			raise NotImplementedError

		# for downstream classification
		classif_q = Bernoulli(logits=mean)
		classif_y = tf.cast(classif_q.sample(), tf.float32)

		# test BSC
		if self.perturb_probs != 0:
			y_hat_prob = tf.nn.sigmoid(mean)
			if self.n_sigbits > 0:
				print('setting first {} bits to be noise-free!'.format(self.n_sigbits))
				adjusted = np.ones((FLAGS.batch_size, self.z_dim)) * self.test_perturb_probs
				adjusted[:, 0:self.n_sigbits] = 1e-6
				adjusted_tensor = tf.constant(adjusted, dtype=tf.float32)
				total_prob = y_hat_prob - (2 * y_hat_prob * adjusted_tensor) + adjusted_tensor
			else:
				total_prob = y_hat_prob - (2 * y_hat_prob * self.test_perturb_probs) + self.test_perturb_probs
			q = Bernoulli(probs=total_prob)
		else:
			print('no additional channel noise; feeding in logits for latent q_phi(z|x) to avoid numerical issues')
			total_prob = tf.nn.sigmoid(mean)
			q = Bernoulli(logits=mean)

		y = tf.cast(q.sample(), tf.float32)
		if dset_name == 'mnist':
			x_reconstr_logits = self.decoder(y, reuse=tf.AUTO_REUSE)
		elif dset_name == 'omniglot':
			# x_reconstr_logits = self.convolutional_32_decoder(y, reuse=tf.AUTO_REUSE)
			x_reconstr_logits = self.decoder(y, reuse=tf.AUTO_REUSE)
		elif dset_name == 'cifar10':
			x_reconstr_logits = self.convolutional_32_decoder(y, reuse=tf.AUTO_REUSE)
		elif dset_name == 'svhn':
			x_reconstr_logits = self.convolutional_32_decoder(y, reuse=tf.AUTO_REUSE)
		elif dset_name == 'celebA':
			x_reconstr_logits = self.complex_decoder(y, reuse=tf.AUTO_REUSE)
		else:
			print('my dataset name is: {} and im lazy'.format(dset_name))
			raise NotImplementedError

		return total_prob, y, classif_y, q, x_reconstr_logits


	# TODO: stefano's suggestion
	def get_collapsed_erasure_stochastic_test_sample(self, x, std=0.1, reuse=False):
		"""
		use collapsed Bernoulli at test time as well
		"""
		print('TEST: implicitly flipping individual bits with probability {}'.format(self.test_perturb_probs))
		dset_name = self.datasource.target_dataset
		if dset_name == 'mnist' or dset_name == 'BinaryMNIST':
			mean = self.encoder(x, reuse=tf.AUTO_REUSE)
		elif dset_name == 'omniglot':
			mean = self.convolutional_32_encoder(x, reuse=tf.AUTO_REUSE)
		elif dset_name == 'cifar10':
			mean = self.cifar10_convolutional_encoder(x, reuse=tf.AUTO_REUSE)
		elif dset_name == 'celebA':
			mean = self.complex_encoder(x, reuse=tf.AUTO_REUSE)
		else:
			print('my dataset name is: {} and im lazy'.format(dset_name))
			raise NotImplementedError

		# for downstream classification
		# classif_q = Bernoulli(logits=mean)
		# classif_y = tf.cast(classif_q.sample(), tf.float32)

		# test BEC
		if self.perturb_probs != 0:
			print('computing probabilities for erasure channel! (test)')
			y_hat_prob = tf.nn.softmax(mean)
			y_hat_prob = tf.clip_by_value(y_hat_prob, 1e-7, 1.-1e-7)

			# construct mask for erasure channel
			mask = np.zeros((2,3))
			mask[0,0] = 1 - self.test_perturb_probs
			mask[0,2] = self.test_perturb_probs
			mask[1,1] = 1 - self.test_perturb_probs
			mask[1,2] = self.test_perturb_probs

			total_prob = tf.reshape(tf.reshape(y_hat_prob, [-1, 2]) @ mask, [-1, self.z_dim, 3])
			total_prob = tf.clip_by_value(total_prob, 1e-7, 1.-1e-7)
			q = Categorical(probs=total_prob)
		else:
			print('Use BSC if there is no channel noise!')
			raise NotImplementedError
			# print('no additional channel noise; feeding in logits for latent q_phi(z|x) to avoid numerical issues')
			# total_prob = tf.nn.sigmoid(mean)
			# q = Bernoulli(logits=mean)

		y = tf.cast(q.sample(), tf.float32)
		print('test y shape: {}'.format(y.get_shape()))

		# decoder
		if dset_name == 'mnist' or dset_name == 'BinaryMNIST':
			x_reconstr_logits = self.decoder(y, reuse=tf.AUTO_REUSE)
		elif dset_name == 'omniglot':
			x_reconstr_logits = self.convolutional_32_decoder(y, reuse=tf.AUTO_REUSE)
		elif dset_name == 'cifar10':
			x_reconstr_logits = self.cifar10_convolutional_decoder(y, reuse=tf.AUTO_REUSE)
		elif dset_name == 'celebA':
			x_reconstr_logits = self.complex_decoder(y, reuse=tf.AUTO_REUSE)
		else:
			print('my dataset name is: {} and im lazy'.format(dset_name))
			raise NotImplementedError

		return total_prob, y, q, x_reconstr_logits
		# return total_prob, y, classif_y, q, x_reconstr_logits


	def get_stochastic_test_sample(self, x, std=0.1, reuse=False):
		# TODO: will have to adjust this as well!

		mean = self.encoder(x, reuse=tf.AUTO_REUSE)
		if self.discrete_relax:
			# gumbel-softmax
			q = RelaxedBernoulli(temperature=self.adj_temp, logits=mean)
			z = q.sample()
		else:
			print('creating stochastic test computation graph')
			q = Bernoulli(logits=mean)
			z = tf.cast(q.sample(), tf.float32)

		# TODO: channel noise
		if self.perturb_probs != 0.:
			print('TEST: flipping individual bits with probability {}'.format(self.perturb_probs))
			# z = random_bit_flip(z, FLAGS.batch_size, self.perturb_probs)
			z = zero_one_random_bit_flip(z, FLAGS.batch_size, self.perturb_probs)
		x_reconstr_logits = self.decoder(z, reuse=tf.AUTO_REUSE)

		return mean, z, x_reconstr_logits


	def train(self, ckpt=None, verbose=True):
		"""
		Trains VAE for specified number of epochs.
		"""
		
		sess = self.sess
		datasource = self.datasource

		if FLAGS.resume:
			if ckpt is None:
				ckpt = tf.train.latest_checkpoint(FLAGS.logdir)
			self.saver.restore(sess, ckpt)

		elif self.transfer:
			log_file = os.path.join(FLAGS.transfer_outdir, 'log.txt')
			if os.path.exists(log_file):
				for line in open(log_file):
					if "Restoring ckpt at epoch" in line:
						ckpt = line.split()[-1]
						break
			if ckpt is None:
				ckpt = tf.train.latest_checkpoint(FLAGS.transfer_logdir)
			self.saver.restore(sess, ckpt)
		else:
			sess.run(self.init_op)
			if not self.learn_A:
				sess.run(self.assign_A_op)

		t0 = time.time()
		train_dataset = datasource.get_dataset('train')
		train_dataset = train_dataset.batch(FLAGS.batch_size)
		train_dataset = train_dataset.shuffle(buffer_size=10000)
		train_iterator = train_dataset.make_initializable_iterator()
		next_train_batch = train_iterator.get_next()

		valid_dataset = datasource.get_dataset('valid')
		valid_dataset = valid_dataset.batch(FLAGS.batch_size)
		valid_iterator = valid_dataset.make_initializable_iterator()
		next_valid_batch = valid_iterator.get_next()

		self.train_writer = tf.summary.FileWriter(FLAGS.outdir + '/train', graph=tf.get_default_graph())
		self.valid_writer = tf.summary.FileWriter(FLAGS.outdir + '/valid', graph=tf.get_default_graph())

		epoch_train_losses = []
		epoch_valid_losses = []
		epoch_save_paths = []
		# keep track of iterations for annealing temperature
		counter = 0  

		for epoch in range(FLAGS.num_epochs):
			sess.run(train_iterator.initializer)
			sess.run(valid_iterator.initializer)
			epoch_train_loss = 0.
			num_batches = 0.
			# counter = 0
			while True:
				try:
					self.training = True
					counter += 1
					if not self.is_binary:
						x = sess.run(next_train_batch)[0]
					else:
						# no labels available for binarized MNIST
						x = sess.run(next_train_batch)
					if self.noisy_mnist:
						# print('training with noisy MNIST...')
						feed_dict = {self.x: (x + np.random.normal(0, 0.5, x.shape)), self.true_x: x}
					else:
						feed_dict = {self.x: x}

					# REINFORCE-style training with VIMCO or vanilla gradient update
					if not self.discrete_relax:
						sess.run([self.discrete_train_op1, self.discrete_train_op2], feed_dict)
					else:
						# this works for both gumbel-softmax
						sess.run(self.train_op, feed_dict)

					batch_loss, train_summary, gs = sess.run([
						self.reconstr_loss, self.summary_op, self.global_step], feed_dict)
					epoch_train_loss += batch_loss

					# self.train_writer.add_summary(train_summary, gs)
					num_batches += 1

				except tf.errors.OutOfRangeError:
					break
			# end of training epoch; adjust temperature here if using Gumbel-Softmax
			if self.discrete_relax:
				if (counter % 1000 == 0) and (counter > 0):
					self.adj_temp = np.maximum(self.tau * np.exp(-self.anneal_rate * counter), self.min_temp)
					print('adjusted temperature to: {}'.format(self.adj_temp))
			# enter validation phase
			if verbose:
				epoch_train_loss /= num_batches
				self.training = False
				# print('in validation phase')
				if not self.is_binary:
					x = sess.run(next_valid_batch)[0]
				else:
					# no labels available for binarized MNIST
					x = sess.run(next_valid_batch)
				if self.noisy_mnist:
					# print('training with noisy MNIST...')
					feed_dict = {self.x: (x + np.random.normal(0, 0.5, x.shape)), self.true_x: x}
				else:
					feed_dict = {self.x: x}

				# save run stats
				epoch_valid_loss, valid_summary, gs = sess.run([self.test_loss, self.summary_op, self.global_step], feed_dict=feed_dict)
				if epoch_train_loss < 0:
					print('Epoch {}, (no sqrt) l2 train loss: {:0.6f}, l2 valid loss: {:0.6f}, time: {}s'. \
				format(epoch+1, epoch_train_loss, np.sqrt(epoch_valid_loss), int(time.time()-t0)))
				else:
					print('Epoch {}, l2 train loss: {:0.6f}, l2 valid loss: {:0.6f}, time: {}s'. \
							format(epoch+1, np.sqrt(epoch_train_loss), np.sqrt(epoch_valid_loss), int(time.time()-t0)))
				sys.stdout.flush()
				save_path = self.saver.save(sess, os.path.join(FLAGS.logdir, 'model.ckpt'), global_step=gs)
				epoch_train_losses.append(epoch_train_loss)
				epoch_valid_losses.append(epoch_valid_loss)
				epoch_save_paths.append(save_path)
		best_ckpt = None
		if verbose:
			min_idx = epoch_valid_losses.index(min(epoch_valid_losses))
			print('Restoring ckpt at epoch', min_idx+1,'with lowest validation error:', epoch_save_paths[min_idx])
			best_ckpt = epoch_save_paths[min_idx]
		return (epoch_train_losses, epoch_valid_losses), best_ckpt

	def test(self, ckpt=None):

		sess = self.sess
		datasource = self.datasource
		self.training = False

		if ckpt is None:
			ckpt = tf.train.latest_checkpoint(FLAGS.logdir)
		
		self.saver.restore(sess, ckpt)

		test_dataset = datasource.get_dataset('test')
		test_dataset = test_dataset.batch(FLAGS.batch_size)
		test_iterator = test_dataset.make_initializable_iterator()
		next_test_batch = test_iterator.get_next()

		test_loss = 0.
		num_batches = 0.
		num_incorrect = 0
		sess.run(test_iterator.initializer)
		while True:
			try:
				if not self.is_binary:
					x, y = sess.run(next_test_batch)
					labels.append(y)
				else:
					# no labels available for binarized MNIST
					x = sess.run(next_test_batch)
				# specify whether to train with noise
				if self.noisy_mnist:
					# print('training with noisy MNIST...')
					feed_dict = {self.x: (x + np.random.normal(0, 0.5, x.shape)), self.true_x: x}
				else:
					feed_dict = {self.x: x}
				# what to save and what to not
				if self.img_dim != 64:
					x_reconstr_logits = sess.run([self.x_reconstr_logits], feed_dict)
				else:
					x_reconstr_logits = sess.run(self.test_x_reconstr_logits, feed_dict)
				batch_test_loss = sess.run(self.test_loss, feed_dict)
				test_loss += batch_test_loss

				# round output of Gaussian decoder to see how many were incorrectly decoded
				rounded = np.round(x_reconstr_logits)
				wrong = np.sum(~np.equal(x, rounded)) 
				num_incorrect += wrong
				num_batches += 1.
			except tf.errors.OutOfRangeError:
				break
		test_loss /= num_batches
		print('L2 squared test loss (per image): {:0.6f}'.format(test_loss))
		print('L2 squared test loss (per pixel): {:0.6f}'.format(test_loss/self.input_dim))

		print('L2 test loss (per image): {:0.6f}'.format(np.sqrt(test_loss)))
		print('L2 test loss (per pixel): {:0.6f}'.format(np.sqrt(test_loss)/self.input_dim))

		return test_loss


	def reconstruct(self, ckpt=None, pkl_file=None):

		import pickle

		sess = self.sess
		datasource = self.datasource

		if ckpt is None:
			ckpt = tf.train.latest_checkpoint(FLAGS.logdir)
		self.saver.restore(sess, ckpt)

		if pkl_file is None:
			test_dataset = datasource.get_dataset('test')
			test_dataset = test_dataset.batch(FLAGS.batch_size)
			test_iterator = test_dataset.make_initializable_iterator()
			next_test_batch = test_iterator.get_next()

			sess.run(test_iterator.initializer)
			if not self.is_binary:
				x = sess.run(next_test_batch)[0]
			else:
				x = sess.run(next_test_batch)
		else:
			with open(pkl_file, 'rb') as f:
				images = pickle.load(f)
			x = np.vstack([images[i] for i in range(10)])
			print(x.shape)
		# grab reconstructions
		if self.noisy_mnist:
			# print('training with noisy MNIST...')
			feed_dict = {self.x: (x + np.random.normal(0, 0.5, x.shape)), self.true_x: x}
		else:
			feed_dict = {self.x: x}
		# grab reconstructions
		x_reconstr_logits = sess.run(self.test_x_reconstr_logits, feed_dict)
		# TODO: rounding values here to get binary MNIST
		# x_reconstr_logits = np.round(x_reconstr_logits)
		print(np.max(x_reconstr_logits), np.min(x_reconstr_logits))
		print(np.max(x), np.min(x))
		
		x_reconstr_logits = np.reshape(x_reconstr_logits, (-1, self.input_dim))
		if self.img_dim == 64:
			x = np.reshape(x, (-1, self.input_dim))
			plot(np.vstack((
				x[0:10], x_reconstr_logits[0:10])), m=10, n=2, px=64, title='reconstructions')
		elif self.img_dim == 32:
			x = np.reshape(x, (-1, self.input_dim))
			plot(np.vstack((
				x[0:10], x_reconstr_logits[0:10])), m=10, n=2, px=32, title='reconstructions')
		elif self.input_dim == 7840 or self.input_dim == 3920:
			x = np.reshape(x, (-1, self.input_dim))
			plot_stitched_mnist(np.vstack((
				x[0:10], x_reconstr_logits[0:10])), m=10, n=20, title='reconstructions')
		else:
			# TODO: edited this
			plot(np.vstack((
				x[0:10], x_reconstr_logits[0:10])), m=10, n=2, title='reconstructions')
		
		with open(os.path.join(FLAGS.outdir, 'reconstr.pkl'), 'wb') as f:
			pickle.dump(x_reconstr_logits, f, pickle.HIGHEST_PROTOCOL)
		return x_reconstr_logits

	def markov_chain(self, ckpt=None):

		sess = self.sess
		datasource = self.datasource

		if ckpt is None:
			ckpt = tf.train.latest_checkpoint(FLAGS.logdir)
		self.saver.restore(sess, ckpt)

		print('initializing with samples from test set...')
		test_dataset = datasource.get_dataset('test')
		test_dataset = test_dataset.batch(FLAGS.batch_size)
		test_iterator = test_dataset.make_initializable_iterator()
		next_test_batch = test_iterator.get_next()

		sess.run(test_iterator.initializer)
		if not self.is_binary:
			x_t = sess.run(next_test_batch)[0]
		else:
			x_t = sess.run(next_test_batch)

		# random initialization with noise
		# print('initializing markov chain with random Gaussian noise...')
		# just initialize 10 samples
		# x_t = np.clip(np.random.normal(
		# 	0., 0.01, 10 * self.input_dim).reshape(-1, self.input_dim), 0., 1.)
		# print('initializing markov chain with random Bernoulli noise...')
		# x_t = np.random.binomial(
			# 1, 0.5, 10 * self.input_dim).reshape(-1, self.input_dim)

		# just get first 10 samples
		samples = [x_t[0:10]]
		for step in range(FLAGS.total_mcmc_steps):
			# whether to train with noise
			if self.noisy_mnist:
				# print('training with noisy MNIST...')
				feed_dict = {self.x: x_t + np.random.normal(0, 0.5, x_t.shape), self.true_x: x_t}
			else:
				feed_dict = {self.x: x_t}

			x_reconstr_mean = sess.run(self.test_x_reconstr_logits, feed_dict)
			x_t_plus_1 = np.clip(np.random.normal(loc=x_reconstr_mean, scale=0.01), 0., 1.)
			x_t = x_t_plus_1

			if (step + 1) % 1000 == 0:
				print('Step', step)
				samples.append(x_t[0:10])

		samples = np.vstack(samples)
		print(samples.shape)
		if self.input_dim != 7840:
			plot(samples, m=10, n=10, title='markov_chain_samples')
		else:
			plot_stitched_mnist(samples, m=10, n=100, title='markov_chain_samples')

		return samples
