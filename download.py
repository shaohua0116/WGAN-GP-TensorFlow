from __future__ import print_function

import os
import subprocess
from glob import glob
import argparse
import tarfile

import h5py
import numpy as np
import scipy.io as sio


parser = argparse.ArgumentParser(description='Download dataset for DCGAN.')
parser.add_argument('--datasets', metavar='N', type=str, nargs='+',
                    choices=['bedroom', 'celeba', 'CIFAR10', 'CIFAR100',
                             'SVHN', 'MNIST', 'Fashion_MNIST'])


def prepare_h5py(train_image, test_image, data_dir, shape=None, rot=False):
    image = np.concatenate((train_image, test_image), axis=0).astype(np.uint8)

    print('Preprocessing data...')

    import progressbar
    bar = progressbar.ProgressBar(
        maxval=100,
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    f = h5py.File(os.path.join(data_dir, 'data.hdf5'), 'w')
    data_id = open(os.path.join(data_dir, 'id.txt'), 'w')
    for i in range(image.shape[0]):

        if i % (image.shape[0]/100) == 0:
            bar.update(i/(image.shape[0]/100))

        grp = f.create_group(str(i))
        data_id.write(str(i)+'\n')
        if shape:
            img = np.reshape(image[i], shape, order='F')
            if rot:
                img = np.rot90(img, k=3)
            grp['image'] = img
        else:
            grp['image'] = image[i]
    bar.finish()
    f.close()
    data_id.close()
    return


def download_mnist(download_path):
    data_dir = os.path.join(download_path, 'mnist')
    if os.path.exists(data_dir):
        print('MNIST was downloaded.')
        return
    else:
        os.mkdir(data_dir)

    data_url = 'http://yann.lecun.com/exdb/mnist/'
    keys = ['train-images-idx3-ubyte.gz', 't10k-images-idx3-ubyte.gz']

    for k in keys:
        url = (data_url+k).format(**locals())
        target_path = os.path.join(data_dir, k)
        cmd = ['curl', url, '-o', target_path]
        print('Downloading ', k)
        subprocess.call(cmd)
        cmd = ['gzip', '-d', target_path]
        print('Unzip ', k)
        subprocess.call(cmd)

    num_mnist_train = 60000
    num_mnist_test = 10000

    fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    train_image = loaded[16:].reshape((num_mnist_train, 28, 28, 1)).astype(np.float)

    fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    test_image = loaded[16:].reshape((num_mnist_test, 28, 28, 1)).astype(np.float)

    prepare_h5py(train_image, test_image, data_dir)

    for k in keys:
        cmd = ['rm', '-f', os.path.join(data_dir, k[:-3])]
        subprocess.call(cmd)


def download_fashion_mnist(download_path):
    data_dir = os.path.join(download_path, 'fashion_mnist')
    if os.path.exists(data_dir):
        print('Fashion MNIST was downloaded.')
        return
    else:
        os.mkdir(data_dir)

    data_url = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
    keys = ['train-images-idx3-ubyte.gz', 't10k-images-idx3-ubyte.gz']

    for k in keys:
        url = (data_url+k).format(**locals())
        target_path = os.path.join(data_dir, k)
        cmd = ['curl', url, '-o', target_path]
        print('Downloading ', k)
        subprocess.call(cmd)
        cmd = ['gzip', '-d', target_path]
        print('Unzip ', k)
        subprocess.call(cmd)

    num_mnist_train = 60000
    num_mnist_test = 10000

    fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    train_image = loaded[16:].reshape((num_mnist_train, 28, 28, 1)).astype(np.float)

    fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    test_image = loaded[16:].reshape((num_mnist_test, 28, 28, 1)).astype(np.float)

    prepare_h5py(train_image, test_image, data_dir)

    for k in keys:
        cmd = ['rm', '-f', os.path.join(data_dir, k[:-3])]
        subprocess.call(cmd)


def svhn_loader(url, path):
    cmd = ['curl', url, '-o', path]
    subprocess.call(cmd)
    m = sio.loadmat(path)
    return np.transpose(m['X'], (3, 0, 1, 2))


def download_svhn(download_path):
    data_dir = os.path.join(download_path, 'svhn')
    if os.path.exists(data_dir):
        print('SVHN was downloaded.')
        return
    else:
        os.mkdir(data_dir)

    print('Downloading SVHN')
    data_url = 'http://ufldl.stanford.edu/housenumbers/train_32x32.mat'
    train_image = svhn_loader(data_url, os.path.join(data_dir, 'train_32x32.mat'))

    data_url = 'http://ufldl.stanford.edu/housenumbers/test_32x32.mat'
    test_image = svhn_loader(data_url, os.path.join(data_dir, 'test_32x32.mat'))

    prepare_h5py(train_image, test_image,  data_dir)

    cmd = ['rm', '-f', os.path.join(data_dir, '*.mat')]
    subprocess.call(cmd)


def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict


def download_cifar(download_path, cifar10=True):
    cifarx = 10 if cifar10 else 100
    data_dir = os.path.join(download_path, 'cifar10' if cifar10 else 'cifar100')
    if os.path.exists(data_dir):
        print('{} was downloaded.'.format('CIFAR10' if cifar10 else 'CIFAR100'))
        return
    else:
        os.mkdir(data_dir)
    data_url = 'https://www.cs.toronto.edu/~kriz/cifar-{}-python.tar.gz'.format(
        cifarx)
    k = 'cifar-{}-python.tar.gz'.format(cifarx)
    target_path = os.path.join(data_dir, k)
    print(target_path)
    cmd = ['curl', data_url, '-o', target_path]
    print('Downloading {}'.format('CIFAR10' if cifar10 else 'CIFAR100'))
    subprocess.call(cmd)
    tarfile.open(target_path, 'r:gz').extractall(data_dir)

    num_cifar_train = 50000
    num_cifar_test = 10000

    if cifar10:
        target_path = os.path.join(data_dir, 'cifar-{}-batches-py'.format(cifarx))
        train_image = []
        for i in range(5):
            fd = os.path.join(target_path, 'data_batch_'+str(i+1))
            dict = unpickle(fd)
            train_image.append(dict['data'])
        train_image = np.reshape(np.stack(train_image, axis=0), [num_cifar_train, 32*32*3])
        fd = os.path.join(target_path, 'test_batch')
        dict = unpickle(fd)
        test_image = np.reshape(dict['data'], [num_cifar_test, 32*32*3])
    else:
        fd = os.path.join(data_dir, 'cifar-100-python', 'train')
        dict = unpickle(fd)
        train_image = dict['data']
        fd = os.path.join(data_dir, 'cifar-100-python', 'test')
        dict = unpickle(fd)
        test_image = dict['data']

    prepare_h5py(train_image, test_image, data_dir, [32, 32, 3], rot=True)

    cmd = ['rm', '-f', os.path.join(
        data_dir, 'cifar-{}-python.tar.gz'.format(cifarx))]
    subprocess.call(cmd)
    if cifar10:
        cmd = ['rm', '-rf', os.path.join(
            data_dir, 'cifar-{}-batches-py'.format(cifarx))]
        subprocess.call(cmd)
    else:
        cmd = ['rm', '-rf', os.path.join(
            data_dir, 'cifar-{}-python'.format(cifarx))]
        subprocess.call(cmd)


def download_file_from_google_drive(id, destination):
    """
    This code is partially borrowed from 
    https://gist.github.com/charlesreid1/4f3d676b33b95fce83af08e4ec261822
    """
    import requests
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def download_celeba(download_path):
    data_dir = os.path.join(download_path, 'celeba')
    if os.path.exists(data_dir):
        if len(glob(os.path.join(data_dir, "*.jpg"))) == 202599:
            print('celeba was downloaded.')
            return
        else:
            print('celeba was downloaded but some files are missing.')
    else:
        os.mkdir(data_dir)

    if os.path.exists(data_dir):
        return
    print('Downloading celeba')
    download_file_from_google_drive(
        "0B7EVK8r0v71pZjFTYXZWM3FlRnM", "celebA.zip")
    print('Unzipping ')
    cmds = [['unzip', 'celebA.zip'], ['rm', 'celebA.zip'],
            ['mv', 'img_align_celeba', 'celeba'], ['mv', 'celeba', download_path]]
    for cmd in cmds:
        subprocess.call(cmd)


def download_bedroom(download_path):
    """
    This code is partially borrowed from
    https://github.com/fyu/lsun/blob/master/download.py
    """
    import lmdb
    category = 'bedroom'
    set_name = 'train'
    data_dir = os.path.join(download_path, category)
    num_total_images = 3033042
    if os.path.exists(data_dir):
        if len(glob(os.path.join(data_dir, "*.webp"))) == 3033042:
            print('Bedroom was downloaded.')
            return
        else:
            print('Bedroom was downloaded but some files are missing.')
    else:
        os.mkdir(data_dir)

    if not os.path.exists("bedroom_train_lmdb/"):
        url = 'http://lsun.cs.princeton.edu/htbin/download.cgi?tag={}' \
            '&category={}&set={}'.format('latest', category, set_name)
        cmd = ['curl', url, '-o', 'bedroom.zip']
        print('Downloading', category, set_name, 'set')
        subprocess.call(cmd)
        print('Unzipping')
        cmds = [['unzip', 'bedroom.zip'], ['rm', 'bedroom.zip']]
        for cmd in cmds:
            subprocess.call(cmd)
    else:
        print("lmdb files are downloaded and unzipped")

    # export the lmdb file to data_dir
    print("Extracting .webp files from lmdb file")
    env = lmdb.open('bedroom_train_lmdb', map_size=1099511627776,
                    max_readers=100, readonly=True)
    count = 0
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for key, val in cursor:
            try:
                image_out_path = os.path.join(data_dir, key + '.webp')
                with open(image_out_path, 'w') as fp:
                    fp.write(val)
            except:
                image_out_path = os.path.join(data_dir, key.decode() + '.webp')
                with open(image_out_path, 'wb') as fp:
                    fp.write(val)
            count += 1
            if count % 1000 == 0:
                print('Finished {}/{} images'.format(count, num_total_images))

    cmd = ['rm', '-rf', 'bedroom_train_lmdb']
    subprocess.call(cmd)


if __name__ == '__main__':
    args = parser.parse_args()
    root_path = './datasets'
    if not os.path.exists(root_path):
        os.mkdir(root_path)

    if 'bedroom' in args.datasets:
        download_bedroom(root_path)
    if 'celeba' in args.datasets:
        download_celeba(root_path)
    if 'CIFAR10' in args.datasets:
        download_cifar(root_path)
    if 'CIFAR100' in args.datasets:
        download_cifar(root_path, cifar10=False)
    if 'SVHN' in args.datasets:
        download_svhn(root_path)
    if 'MNIST' in args.datasets:
        download_mnist(root_path)
    if 'Fashion_MNIST' in args.datasets:
        download_fashion_mnist(root_path)
