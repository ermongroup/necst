# TODO: automatically crops and aligns image to be size 64 x 64

import os
import h5py
import numpy as np
from PIL import Image


IMG_DIR = '/atlas/u/kechoi/csgm-private/datasets/celebA/img_align_celeba/'
ATTRIBUTES_FILE = '/atlas/u/kechoi/csgm-private/datasets/celebA/list_attr_celeba.txt'
NUM_EXAMPLES = 202599
TRAIN_STOP = 162770
VALID_STOP = 182637


# with labels
def convert_celebA(output_path, name):

	image_shape = (64, 64)
	# read in labels
	targets = (np.loadtxt(os.path.join(output_path, ATTRIBUTES_FILE), dtype='int32', skiprows=2, usecols=tuple(range(1, 41))) +
    1) / 2
	targets = np.reshape(targets[:, 20], (-1, 1))
    # specifically look at gender --> 0: female, 0: male
    # TODO: another one you could do is smiling: targets[:, 31]

	h5file = h5py.File(os.path.join(output_path, name), mode='w')
	print('writing h5 record file to {}...'.format(h5file))
	
	# train set
	h5file.create_dataset('train', [TRAIN_STOP, 64, 64, 3], dtype='uint8')
	h5file.create_dataset('trainlabels', [TRAIN_STOP, 1], dtype='uint8')

	train_dataset = h5file['train']
	targets_dataset = h5file['trainlabels']
	for i in np.arange(0, TRAIN_STOP):
		img_name = IMG_DIR + '{:06d}.jpg'.format(i + 1)
		# resize and crop to [64, 64, 3]
		new_img = Image.open(img_name).resize((64, 78), Image.ANTIALIAS).crop((0, 7, 64, 64 + 7))
		train_dataset[i] = np.asarray(new_img).astype(np.uint8)
		targets_dataset[i] = targets[i]
	print('finished converting training set')

	# validation set
	h5file.create_dataset('valid', [VALID_STOP - TRAIN_STOP, 64, 64, 3], dtype='uint8')
	h5file.create_dataset('validlabels', [VALID_STOP - TRAIN_STOP, 1], dtype='uint8')

	valid_dataset = h5file['valid']
	valid_targets_dataset = h5file['validlabels']
	for i in np.arange(0, VALID_STOP-TRAIN_STOP):
		img_name = IMG_DIR + '{:06d}.jpg'.format((i + TRAIN_STOP) + 1)
		# resize and crop to [64, 64, 3]
		new_img = Image.open(img_name).resize((64, 78), Image.ANTIALIAS).crop((0, 7, 64, 64 + 7))
		valid_dataset[i] = np.asarray(new_img).astype(np.uint8)
		valid_targets_dataset[i] = targets[i]
	print('finished converting validation set')
	
	# test set
	h5file.create_dataset('test', [NUM_EXAMPLES - VALID_STOP, 64, 64, 3], dtype='uint8')
	h5file.create_dataset('testlabels', [NUM_EXAMPLES - VALID_STOP, 1], dtype='uint8')

	test_dataset = h5file['test']
	test_targets_dataset = h5file['testlabels']

	for i in np.arange(0, NUM_EXAMPLES-VALID_STOP):
	    img_name = IMG_DIR + '{:06d}.jpg'.format((i + VALID_STOP) + 1)
	    new_img = Image.open(img_name).resize((64, 78), Image.ANTIALIAS).crop((0, 7, 64, 64 + 7))
	    test_dataset[i] = np.asarray(new_img).astype(np.uint8)
	    test_targets_dataset[i] = targets[i]
	print('finished converting test set')
	
	# write to record file
	h5file.flush()
	h5file.close()


# def convert_celebA(output_path, name):

# 	image_shape = (64, 64)
# 	# read in labels
# 	targets = (np.loadtxt(os.path.join(directory, ATTRIBUTES_FILE), dtype='int32', skiprows=2, usecols=tuple(range(1, 41))) +
#     1) / 2
#     # specifically look at gender --> 0: female, 0: male
#     # TODO: another one you could do is smiling: targets[:, 31]
#     targets = np.reshape(targets[:, 20], (-1, 1))

# 	h5file = h5py.File(os.path.join(output_path, name), mode='w')
# 	print('writing h5 record file to {}...'.format(h5file))
	
# 	# train set
# 	h5file.create_dataset('train', [TRAIN_STOP, 64, 64, 3], dtype='uint8')
# 	train_dataset = h5file['train']
# 	for i in np.arange(0, TRAIN_STOP):
# 		img_name = IMG_DIR + '{:06d}.jpg'.format(i + 1)
# 		# resize and crop to [64, 64, 3]
# 		new_img = Image.open(img_name).resize((64, 78), Image.ANTIALIAS).crop((0, 7, 64, 64 + 7))
# 		train_dataset[i] = np.asarray(new_img).astype(np.uint8)
# 	print('finished converting training set')

# 	# validation set
# 	h5file.create_dataset('valid', [VALID_STOP - TRAIN_STOP, 64, 64, 3], dtype='uint8')
# 	valid_dataset = h5file['valid']
# 	for i in np.arange(0, VALID_STOP-TRAIN_STOP):
# 		img_name = IMG_DIR + '{:06d}.jpg'.format((i + TRAIN_STOP) + 1)
# 		# resize and crop to [64, 64, 3]
# 		new_img = Image.open(img_name).resize((64, 78), Image.ANTIALIAS).crop((0, 7, 64, 64 + 7))
# 		valid_dataset[i] = np.asarray(new_img).astype(np.uint8)
# 	print('finished converting validation set')
	
# 	# test set
# 	h5file.create_dataset('test', [NUM_EXAMPLES - VALID_STOP, 64, 64, 3], dtype='uint8')
# 	test_dataset = h5file['test']
# 	for i in np.arange(0, NUM_EXAMPLES-VALID_STOP):
# 	    img_name = IMG_DIR + '{:06d}.jpg'.format((i + VALID_STOP) + 1)
# 	    new_img = Image.open(img_name).resize((64, 78), Image.ANTIALIAS).crop((0, 7, 64, 64 + 7))
# 	    test_dataset[i] = np.asarray(new_img).astype(np.uint8)
# 	print('finished converting test set')
	
# 	# write to record file
# 	h5file.flush()
# 	h5file.close()


def main():
    convert_celebA('/atlas/u/kechoi/csgm-private/datasets/celebA/', 'REAL_gender_celeba_aligned_cropped.hdf5')


if __name__ == '__main__':
    main()
