import os
import numpy as np


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('out_dir', type=str, help='where to save outputs')
	parser.add_argument('n_bits', type=int, help='number of random bits')
	args = parser.parse_args()

	# set random seed
	np.random.seed(1234)

	if not os.path.isdir(args.out_dir):
	    os.makedirs(args.out_dir)

	# generate 5K train, 1K valid, 1K test of <args.n_bits>-bit signals
	train = np.reshape(np.random.randint(0, 2, 5000*args.n_bits), (5000, -1))
	valid = np.reshape(np.random.randint(0, 2, 1000*args.n_bits), (1000, -1))
	test = np.reshape(np.random.randint(0, 2, 1000*args.n_bits), (1000, -1))

	# save data
	np.save(os.path.join(args.out_dir, 'random_bits_train.npy'), train)
	np.save(os.path.join(args.out_dir, 'random_bits_valid.npy'), valid)
	np.save(os.path.join(args.out_dir, 'random_bits_test.npy'), test)
