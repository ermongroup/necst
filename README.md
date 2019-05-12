# Neural Joint-Source Channel Coding

This repo contains a reference implementation for NECST as described in the paper:
> Neural Joint-Source Channel Coding </br>
> [Kristy Choi](http://kristychoi.com/), [Kedar Tatwawadi](https://web.stanford.edu/~kedart/), [Aditya Grover](http://aditya-grover.github.io/), [Tsachy Weissman](https://web.stanford.edu/~tsachy/), [Stefano Ermon](https://cs.stanford.edu/~ermon/) </br>
> International Conference on Machine Learning (ICML), 2019. </br>
> Paper: https://arxiv.org/abs/1811.07557 </br>


## Requirements
The codebase is implemented in Python 3.6 and Tensorflow. To install the necessary dependencies, run:
```
pip3 install -r requirements.txt

```

## Datasets
A set of scripts for data pre-processing are included in the directory `./data_setup`. Relevant files for 
The NECST model operates over Tensorflow [TFRecords](https://www.tensorflow.org/tutorials/load_data/tf_records). A few points to note:

1. Raw data files for MNIST and BinaryMNIST can be downloaded using `data_setup/download.py`. CelebA files can be downloaded using `data_setup/celebA_download.py`. CIFAR10 can be downloaded (with tfrecords automatically generated) using `data_setup/generate_cifar10_tfrecords.py`. All other data files (Omniglot, SVHN) must be downloaded separately.
2. Omniglot and CelebA should be converted into `.hdf5` format using `data_setup/convert_celebA_h5.py` and `data_setup/convert_omniglot_h5.py` respectively.
3. Random {0,1} bits can be generated using `data_setup/gen_random_bits.py`.
4. After this step, tfrecords must be generated using: `data_setup/convert_to_records.py` before running the model.

## Options
Training the NECST model takes a set of command line arguments in the `main.py` script. The most relevant ones are listed below:
```
--datasource (STRING):    one of [mnist, BinaryMNIST, random, omniglot, celebA, svhn, cifar10]
--is_binary (BOOL):       whether or not the data is binary {0,1}, e.g. BinaryMNIST
--vimco_samples (INT):    number of samples to use for VIMCO
--channel_model (STRING): BSC/BEC
--noise (FLOAT):          channel noise level during training
--test_noise (FLOAT):     channel noise level at TEST time
--n_epochs (INT):         number of training epochs
--batch_size (INT):       size of minibatch
--lr (FLOAT):             learning rate of optimizer
--optimizer (STRING):     one of [adam, sgd]
--dech_arch (STRING):     comma-separated decoder architecture
--enc_arch (STRING):      comma-separated encoder architecture
--reg_param (FLOAT):      regularization for encoder architecture
```

## Examples
Training a 100-bit NECST model with BSC noise = 0.1 on BinaryMNIST:
```
python3 main.py --datadir=./data --datasource=BinaryMNIST --channel_model=bsc --noise=0.1 --test_noise=0.1 --n_bits=100 --is_binary=True
```
Training a 1000-bit NECST model with BSC noise = 0.2 on CelebA:
```
python3 main.py --datadir=./data --datasource=celebA --channel_model=bsc --noise=0.2 --test_noise=0.2 --n_bits=1000
```

## Citing
If you find NECST useful in your research, please consider citing the following paper:

```
@article{choi2018necst,
  title={Neural Joint Source-Channel Coding},
  author={Choi, Kristy and Tatwawadi, Kedar and Grover, Aditya and Weissman, Tsachy and Ermon, Stefano},
  journal={arXiv preprint arXiv:1811.07557},
  year={2018}
}
```
