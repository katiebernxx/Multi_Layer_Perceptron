'''
Preprocessing data in STL-10 image dataset
Katie Bernard
'''
import numpy as np
import load_stl10_dataset


def preprocess_stl(imgs, labels):
    '''Preprocesses stl image data for training by a MLP neural network

    Parameters:
    ----------
    imgs: unint8 ndarray  [0, 255]. shape=(Num imgs, height, width, RGB color chans)

    Returns:
    ----------
    imgs: float64 ndarray [0, 1]. shape=(Num imgs N,)
    Labels: int ndarray. shape=(Num imgs N,). Contains int-coded class values 0,1,...,9
    '''
    imgs = imgs.astype(np.float64)
    imgs = imgs.reshape(imgs.shape[0], -1)

    means = imgs.mean(axis=0)
    stdevs = imgs.std(axis=0)
    imgs = (imgs - means) / stdevs
    
    labels = labels - 1
     
    return imgs, labels



def create_splits(data, y, n_train_samps=3500, n_test_samps=500, n_valid_samps=500, n_dev_samps=500):
    '''Divides the dataset up into train/test/validation/development "splits" (disjoint partitions)

    Parameters:
    ----------
    data: float64 ndarray. Image data. shape=(Num imgs, height*width*chans)
    y: ndarray. int-coded labels.

    Returns:
    ----------
    None if error
    x_train (training samples),
    y_train (training labels),
    x_test (test samples),
    y_test (test labels),
    x_val (validation samples),
    y_val (validation labels),
    x_dev (development samples),
    y_dev (development labels)

    '''

    x_train, y_train = data[:n_train_samps], y[:n_train_samps]
    x_test, y_test = data[n_train_samps:n_train_samps + n_test_samps], y[n_train_samps:n_train_samps + n_test_samps]
    x_val, y_val = data[n_train_samps + n_test_samps:n_train_samps + n_test_samps + n_valid_samps], y[n_train_samps + n_test_samps:n_train_samps + n_test_samps + n_valid_samps]
    x_dev, y_dev = data[n_train_samps + n_test_samps + n_valid_samps:], y[n_train_samps + n_test_samps + n_valid_samps:]

    if n_train_samps + n_test_samps + n_valid_samps + n_dev_samps != len(data):
        samps = n_train_samps + n_test_samps + n_valid_samps + n_dev_samps
        print(f'Error! Num samples {samps} does not equal num images {len(data)}!')
        return

    return x_train, y_train, x_test, y_test, x_val, y_val, x_dev, y_dev


def load_stl10(n_train_samps=3500, n_test_samps=500, n_valid_samps=500, n_dev_samps=500):
    '''Automates the process of:
    - loading in the STL-10 dataset and labels
    - preprocessing
    - creating the train/test/validation/dev splits.

    Returns:
    ----------
    None if error
    x_train (training samples),
    y_train (training labels),
    x_test (test samples),
    y_test (test labels),
    x_val (validation samples),
    y_val (validation labels),
    x_dev (development samples),
    y_dev (development labels)
    '''

    stl_imgs, stl_labels = load_stl10_dataset.load()
    stl_imgs, stl_labels = preprocess_stl(stl_imgs, stl_labels)
    x_train, y_train, x_test, y_test, x_val, y_val, x_dev, y_dev = create_splits(stl_imgs, stl_labels, n_train_samps, n_test_samps, n_valid_samps, n_dev_samps)
    return x_train, y_train, x_test, y_test, x_val, y_val, x_dev, y_dev
