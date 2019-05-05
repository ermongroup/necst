# TODO: automatically crops and aligns image to be size 64 x 64
import os
import h5py
import numpy as np
from PIL import Image


NUM_EXAMPLES = 202599
TRAIN_STOP = 162770
VALID_STOP = 182637


def convert_celebA(img_dir, output_path, name):

	image_shape = (64, 64)
	# NOTE: currently not working with any attributes!
	h5file = h5py.File(os.path.join(output_path, name), mode='w')
	print('writing h5 record file to {}...'.format(h5file))
	
	# train set
	h5file.create_dataset('train', [TRAIN_STOP, 64, 64, 3], dtype='uint8')
	train_dataset = h5file['train']
	for i in np.arange(0, TRAIN_STOP):
		img_name = img_dir + '{:06d}.jpg'.format(i + 1)
		# resize and crop to [64, 64, 3]
		new_img = Image.open(img_name).resize((64, 78), Image.ANTIALIAS).crop((0, 7, 64, 64 + 7))
		train_dataset[i] = np.asarray(new_img).astype(np.uint8)
	print('finished converting training set')

	# validation set
	h5file.create_dataset('valid', [VALID_STOP - TRAIN_STOP, 64, 64, 3], dtype='uint8')
	valid_dataset = h5file['valid']
	for i in np.arange(0, VALID_STOP-TRAIN_STOP):
		img_name = img_dir + '{:06d}.jpg'.format((i + TRAIN_STOP) + 1)
		# resize and crop to [64, 64, 3]
		new_img = Image.open(img_name).resize((64, 78), Image.ANTIALIAS).crop((0, 7, 64, 64 + 7))
		valid_dataset[i] = np.asarray(new_img).astype(np.uint8)
	print('finished converting validation set')
	
	# test set
	h5file.create_dataset('test', [NUM_EXAMPLES - VALID_STOP, 64, 64, 3], dtype='uint8')
	test_dataset = h5file['test']
	for i in np.arange(0, NUM_EXAMPLES-VALID_STOP):
	    img_name = img_dir + '{:06d}.jpg'.format((i + VALID_STOP) + 1)
	    new_img = Image.open(img_name).resize((64, 78), Image.ANTIALIAS).crop((0, 7, 64, 64 + 7))
	    test_dataset[i] = np.asarray(new_img).astype(np.uint8)
	print('finished converting test set')
	
	# write to record file
	h5file.flush()
	h5file.close()


def main(args):
	convert_celebA(args.img_dir, args.out_dir, 'celeba_aligned_cropped.hdf5')


if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('img_dir', type=str, help='path to celebA img_align_celeba/ directory with pictures')
	parser.add_argument('out_dir', type=str, help='where to save outputs')
	args = parser.parse_args()

	main(args)
