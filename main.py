import numpy as np 
import tensorflow as tf 
from utils import *
from datasource import Datasource

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.python.platform import flags
FLAGS = flags.FLAGS

# File options
flags.DEFINE_string('datadir', '../../datasets/', 'directory for datasets')
flags.DEFINE_string('datasource', 'mnist', 'mnist/BinaryMNIST/random/omniglot/binary_omniglot/celebA/svhn/cifar10/mnist2omniglot/omniglot2mnist')
flags.DEFINE_string('logdir', './models/', 'directory to save checkpoints, events files')
flags.DEFINE_string('outdir', './results/', 'directory to save samples, final results')
flags.DEFINE_bool('resume', False, 'resume training if there is a model available')
flags.DEFINE_bool('train', True, 'True to train.')
flags.DEFINE_bool('test', True, 'True to test.')
# flags.DEFINE_bool('is_discrete', False, 'True if dataset is discrete, false otherwise.')
flags.DEFINE_string('ckpt', None, 'ckpt to load if resume is True. Defaults (None) to latest ckpt in logdir')
flags.DEFINE_string('exp_id', '0', 'exp_id appended to logdir and outdir')
flags.DEFINE_string('gpu_id', '0', 'gpu id options')
flags.DEFINE_bool('dump', True, 'Dumps to log.txt if True')

# Probabilistic latent bit representation
flags.DEFINE_bool('is_binary', False, 'True if dataset is binary, false otherwise.')
flags.DEFINE_integer('vimco_samples', 5, 'number of VIMCO samples to use during training')
flags.DEFINE_bool('stochastic_discrete_latent', True, 'use REINFORCE-type training to train')
flags.DEFINE_bool('learn_covariance', False, 'learn covariance of Gaussian decoder')
flags.DEFINE_bool('discrete_relax', False, 'use gumbel-softmax trick to learn discrete y')
flags.DEFINE_float('tau', 1.0, 'initial temperature for gumbel-softmax trick')

# Noise specifications
flags.DEFINE_bool('noisy_mnist', False, 'specify whether to train UAE with noisy MNIST')
flags.DEFINE_string('channel_model', 'bsc', 'bsc/bec')
flags.DEFINE_float('perturb_probs', 0., 'specify proportion of entires to corrupt in z')
flags.DEFINE_float('test_perturb_probs', 0., 'specify proportion of entries to corrupt in z AT TEST TIME.')
flags.DEFINE_integer('n_sigbits', 0, 'number of first X significant bits to make noise-free')

# Training options
flags.DEFINE_integer('num_epochs', 500, 'number of training epochs')
flags.DEFINE_integer('batch_size', 100, 'number of datapoints per batch')
flags.DEFINE_float('lr', 0.001, 'learning rate for the model')
flags.DEFINE_string('optimizer', 'adam', 'sgd, adam, momentum')
flags.DEFINE_integer('log_interval', 500, 'training steps after which summary and checkpoints dumped')
flags.DEFINE_integer('num_samples', 16, 'number of samples to generate')

# Model options
flags.DEFINE_string('model', 'uae', 'uae/cuae')
flags.DEFINE_string('activation', 'relu', 'sigmoid/tanh/softplus/leakyrelu/relu')
flags.DEFINE_integer('seed', 0, 'random seed for initializing model parameters')
flags.DEFINE_string('dec_arch', '500,500', 'comma-separated decoder architecture')
flags.DEFINE_string('enc_arch', '500', 'comma-separated encoder architecture')
flags.DEFINE_integer('num_measurements', 10, 'number of measurements')
flags.DEFINE_float('noise_std', 0.1, 'std. of noise')
flags.DEFINE_float('reg_param', 0.0001, 'regularization for encoder')
flags.DEFINE_bool('learn_A', True, 'learns the measurement matrix if True')
flags.DEFINE_string('A_file', './results/0/A.npy', 'file for precomputed measurement matrix')
flags.DEFINE_bool('non_linear_act', True, 'nonlinear activation on final layer of encoder if True')

flags.DEFINE_bool('transfer', False, 'restores transfer ckpt if true')
flags.DEFINE_string('transfer_outdir', './results/', 'directory containing log.txt for source domain')
flags.DEFINE_string('transfer_logdir', './models/', 'directory containing ckpts for source domain')
flags.DEFINE_integer('total_mcmc_steps', 9000, 'number of mcmc steps')
flags.DEFINE_string('pkl_file', None, 'pkl file for reconstruction')

# sanity check
flags.DEFINE_bool('combine', False, 'whether to combine omniglot and mnist to debug')
flags.DEFINE_string('datasource1', 'mnist', 'first datasource for joint training')
flags.DEFINE_string('datasource2', 'omniglot', 'second datasource for joint training')

# experiment using VAE
flags.DEFINE_bool('vae', False, 'specify True if UAE will be run as a VAE instead')


def process_flags():
	"""
	Processes easy-to-specify cmd line FLAGS to appropriate syntax
	"""
	FLAGS.optimizer = get_optimizer_fn(FLAGS.optimizer)
	FLAGS.activation = get_activation_fn(FLAGS.activation)
	
	if FLAGS.dec_arch == '':
		FLAGS.dec_arch = []
	else:
		FLAGS.dec_arch = list(map(int, FLAGS.dec_arch.split(',')))
	
	if FLAGS.enc_arch == '':
		FLAGS.enc_arch = []
	else:
		FLAGS.enc_arch = list(map(int, FLAGS.enc_arch.split(',')))

	return

def main():
	"""
	Runs the ML loop. Preprocesses data, trains model, along with regular validation and testing.
	"""	
	
	os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu_id)
	os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
	subpath = 'noise_' + str(FLAGS.noise_std) 
	FLAGS.logdir = os.path.join(FLAGS.logdir, FLAGS.datasource, subpath, FLAGS.exp_id)
	FLAGS.outdir = os.path.join(FLAGS.outdir, FLAGS.datasource, subpath, FLAGS.exp_id)

	if FLAGS.transfer:
		source = FLAGS.datasource.split("2")[0]
		FLAGS.transfer_logdir = os.path.join(FLAGS.transfer_logdir, source, subpath, FLAGS.exp_id)
		FLAGS.transfer_outdir = os.path.join(FLAGS.transfer_outdir, source, subpath, FLAGS.exp_id)
	
	if not os.path.exists(FLAGS.logdir):
		os.makedirs(FLAGS.logdir)
	if not os.path.exists(FLAGS.outdir):
		os.makedirs(FLAGS.outdir)

	import json
	with open(os.path.join(FLAGS.outdir, 'config.json'), 'w') as fp:
	    json.dump(tf.app.flags.FLAGS.flag_values_dict(), fp, indent=4, separators=(',', ': '))

	if FLAGS.dump:
		import sys
		sys.stdout = open(os.path.join(FLAGS.outdir, 'log.txt'), 'w')

	process_flags()
	gpu_options = tf.GPUOptions(allow_growth=True)
	sess = tf.Session(config=tf.ConfigProto(
		gpu_options=gpu_options, allow_soft_placement=True))
	
	# if FLAGS.combine:
	# 	print('FLAGS.combine is True; running model with both mnist and celebA')
	# 	datasource1 = Datasource(sess)  # this should be mnist
	# 	datasource2 = Datasource(sess)  # this should be celebA

	# 	model_class = load_dynamic(FLAGS.model.upper(), FLAGS.model)
	# 	model = model_class(sess, [datasource1, datasource2])
	# else:
	datasource = Datasource(sess)
	model_class = load_dynamic(FLAGS.model.upper(), FLAGS.model)
	model = model_class(sess, datasource)

	# run computational graph
	best_ckpt = FLAGS.ckpt
	if best_ckpt is not None:
		print('resuming ckpt supplied; restoring model from {}'.format(best_ckpt))
	# best_ckpt = None
	if FLAGS.train:
		learning_curves, best_ckpt = model.train(ckpt=best_ckpt)
	
	if FLAGS.test:
		if best_ckpt is None:
			log_file = os.path.join(FLAGS.outdir, 'log.txt')
			if os.path.exists(log_file):
				for line in open(log_file):
					if "Restoring ckpt at epoch" in line:
						best_ckpt = line.split()[-1]
						break
		model.test(ckpt=best_ckpt)
		# model.get_A(ckpt=best_ckpt, save_file=os.path.join(FLAGS.outdir, 'A.npy'))
		model.reconstruct(ckpt=best_ckpt, pkl_file=FLAGS.pkl_file)
		# model.markov_chain(ckpt=best_ckpt)

if __name__ == "__main__":
	main()

