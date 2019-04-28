import numpy as np

# set random seed
np.random.seed(1234)

# generate 5K train, 1K valid, 1K test of 100-bit signals
train = np.reshape(np.random.randint(0, 2, 5000*100), (5000, -1))
valid = np.reshape(np.random.randint(0, 2, 1000*100), (1000, -1))
test = np.reshape(np.random.randint(0, 2, 1000*100), (1000, -1))

# save data
np.save('/atlas/u/kechoi/csgm-private/datasets/random/random_bits_train.npy', train)
np.save('/atlas/u/kechoi/csgm-private/datasets/random/random_bits_valid.npy', valid)
np.save('/atlas/u/kechoi/csgm-private/datasets/random/random_bits_test.npy', test)
