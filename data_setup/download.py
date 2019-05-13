"""
This file is from https://github.com/carpedm20/DCGAN-tensorflow/blob/master/download.py
It comes with the following License : https://github.com/carpedm20/DCGAN-tensorflow/blob/master/LICENSE

Downloads the following:
- Celeb-A dataset
- LSUN dataset
- MNIST dataset
"""
# pylint: skip-file
import os
import sys
import gzip
import json
import shutil
import zipfile
import argparse
import subprocess
from six.moves import urllib


parser = argparse.ArgumentParser(description='Download dataset for NECST.')
parser.add_argument('datasets', metavar='N', type=str, nargs='+', choices=['mnist', 'BinaryMNIST'],
           help='name of dataset to download [mnist, BinaryMNIST]')

def download(url, dirpath):
  filename = url.split('/')[-1]
  filepath = os.path.join(dirpath, filename)
  u = urllib.request.urlopen(url)
  f = open(filepath, 'wb')
  filesize = int(u.headers["Content-Length"])
  print("Downloading: %s Bytes: %s" % (filename, filesize))

  downloaded = 0
  block_sz = 8192
  status_width = 70
  while True:
    buf = u.read(block_sz)
    if not buf:
      print('')
      break
    else:
      print('', end='\r')
    downloaded += len(buf)
    f.write(buf)
    status = (("[%-" + str(status_width + 1) + "s] %3.2f%%") %
      ('=' * int(float(downloaded) / filesize * status_width) + '>', downloaded * 100. / filesize))
    print(status, end='')
    sys.stdout.flush()
  f.close()
  return filepath

def unzip(filepath):
  print("Extracting: " + filepath)
  dirpath = os.path.dirname(filepath)
  with zipfile.ZipFile(filepath) as zf:
    zf.extractall(dirpath)
  os.remove(filepath)

def download_mnist(dirpath):
  data_dir = os.path.join(dirpath, 'mnist')
  if os.path.exists(data_dir):
    print('Found MNIST - skip')
    return
  else:
    os.mkdir(data_dir)
  url_base = 'http://yann.lecun.com/exdb/mnist/'
  file_names = ['train-images-idx3-ubyte.gz',
                'train-labels-idx1-ubyte.gz',
                't10k-images-idx3-ubyte.gz',
                't10k-labels-idx1-ubyte.gz']
  for file_name in file_names:
    url = (url_base+file_name).format(**locals())
    print(url)
    out_path = os.path.join(data_dir,file_name)
    cmd = ['curl', url, '-o', out_path]
    print('Downloading ', file_name)
    subprocess.call(cmd)
    # cmd = ['gzip', '-d', out_path]
    # print('Decompressing ', file_name)
    # subprocess.call(cmd)

def download_binary_mnist(dirpath):
  """
  this is from yburda's iwae github repo for downloading binary MNIST
  """
  print('Downloading binarized MNIST datasets (these do not include digit labels)...') 
  data_dir = os.path.join(dirpath, 'BinaryMNIST')
  if os.path.exists(data_dir):
    print('Found BinaryMNIST - skip')
    return
  else:
    os.mkdir(data_dir)
  
  subdatasets = ['train', 'valid', 'test']
  for subdataset in subdatasets:
    filename = 'binarized_mnist_{}.amat'.format(subdataset)
    url = 'http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_{}.amat'.format(subdataset)
    local_filename = os.path.join(data_dir, filename)
    urllib.request.urlretrieve(url, local_filename)

def prepare_data_dir(path = './data'):
  if not os.path.exists(path):
    os.mkdir(path)


if __name__ == '__main__':
  args = parser.parse_args()
  prepare_data_dir()

  if 'mnist' in args.datasets:
    download_mnist('./data')
  if 'BinaryMNIST' in args.datasets:
    download_binary_mnist('./data')
