import os
import h5py
import numpy as np
from scipy.io import loadmat

NUM_EXAMPLES = 32415
TRAIN_STOP =  23845
VALID_STOP = 24345


def prepare_h5(output_path):
	"""
	creates datasets to write to in h5 file
	"""
	image_shape = (28, 28)
	h5file = h5py.File(output_path, mode='w')
	
	starts = [0, TRAIN_STOP, VALID_STOP]
	ends = [TRAIN_STOP, VALID_STOP, NUM_EXAMPLES]

	feature_labels = ['train', 'valid', 'test']
	target_labels = ['trainlabels', 'validlabels', 'testlabels']
	more_labels = ['trainlabels2', 'validlabels2', 'testlabels2']

	for s, e, f_label, t_label, m_label in zip(starts, ends, feature_labels, target_labels, more_labels):
		print('feature label name: {}'.format(f_label))
		print('target label 1 name: {}'.format(t_label))
		print('target label 2 name: {}'.format(m_label))
		print('dataset size: {}'.format(e-s))
		# "target"
		targets_dataset = h5file.create_dataset(
			t_label, (e-s, 50), dtype='uint8')
		# "targetchar"
		more_targets_dataset = h5file.create_dataset(
			m_label, (e-s, 1), dtype='uint8')
		# images
		features_dataset = h5file.create_dataset(
		    f_label, (e-s, 1) + image_shape, dtype='float64')

	return h5file


def convert_omniglot(image_file, output_directory, output_filename='omniglot.hdf5'):

	output_path = os.path.join(output_directory, output_filename)
	print(output_path)
	h5file = prepare_h5(output_path)

	# load data
	data = loadmat(image_file)

	# begin partitioning
	# training set
	features_dataset = h5file['train']
	targets_dataset = h5file['trainlabels']
	more_targets_dataset = h5file['trainlabels2']
	for i in np.arange(0, TRAIN_STOP):
		# casting for train, valid, and test to save memory
		features_dataset[i] = np.round(np.reshape(data['data'].T[i], (1, 1, 28, 28)))
		# features_dataset[i] = np.reshape(data['data'].T[i], (1, 1, 28, 28))
		targets_dataset[i] = data['target'].T[i]
		more_targets_dataset[i] = data['targetchar'].T[i]
	# validation set
	valid_features_dataset = h5file['valid']
	valid_targets_dataset = h5file['validlabels']
	valid_more_targets_dataset = h5file['validlabels2']
	for i in np.arange(0, VALID_STOP-TRAIN_STOP):
		valid_features_dataset[i] = np.round(np.reshape(data['data'].T[i], (1, 1, 28, 28)))
		# valid_features_dataset[i] = np.reshape(data['data'].T[i], (1, 1, 28, 28))
		valid_targets_dataset[i] = data['target'].T[i]
		valid_more_targets_dataset[i] = data['targetchar'].T[i]
	# test set
	test_features_dataset = h5file['test']
	test_targets_dataset = h5file['testlabels']
	test_more_targets_dataset = h5file['testlabels2']
	for i in np.arange(0, NUM_EXAMPLES-VALID_STOP):
		test_features_dataset[i] = np.round(np.reshape(data['testdata'].T[i], (1, 1, 28, 28)))
		# test_features_dataset[i] = np.reshape(data['testdata'].T[i], (1, 1, 28, 28))
		test_targets_dataset[i] = data['testtarget'].T[i]
		test_more_targets_dataset[i] = data['testtargetchar'].T[i]
	# write to record file
	h5file.flush()
	h5file.close()

	return (output_path,)


def main(args):
	# TODO: you'll probably want to make that a command-line argument
	convert_omniglot(args.image_file, args.out_dir)


if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('image_file', type=str, help='path to omniglot chardata.mat')
	parser.add_argument('out_dir', type=str, help='where to save outputs')
	args = parser.parse_args()

	if not os.path.isdir(args.out_dir):
		os.makedirs(args.out_dir)

	main(args)