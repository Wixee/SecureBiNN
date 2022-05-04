import os
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn.datasets as sklearn_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from PIL import Image


def shuffle_dataset(X, y, random_state):
    np.random.RandomState(seed=random_state)
    idx = np.random.shuffle(list(range(len(X))))
    return X[idx][0], y[idx][0]


################ Datasets ##############################

def load_dataset(dataset_name, flatten=False, scale=True, dtype='float32', archive_path=None):

    dataset_name = dataset_name.lower()

    SUPPORTED_DATASETS = {
        'mnist',
        'cifar10',
        'fashion_mnist',
        'breast_cancer',
        'diabetes',
        'indian_liver_patient_records',
        'malaria_cell_images'
    }

    assert dataset_name in SUPPORTED_DATASETS, 'The data set must be one of {}.'.format(
        SUPPORTED_DATASETS)

    del SUPPORTED_DATASETS

    if archive_path is None:
        archive_path = Path('archive')
    if type(archive_path) is str:
        archive_path = Path(archive_path)

    if dataset_name == 'mnist':
        shuffle = False
        random_state = 0
        (X_train, y_train), (X_test, y_test) = load_mnist(
            flatten=flatten, scale=scale, shuffle=shuffle, dtype=dtype, random_state=random_state)
    elif dataset_name == 'cifar10':
        shuffle = False
        random_state = 0
        (X_train, y_train), (X_test, y_test) = load_cifar10(
            flatten=flatten, scale=scale, shuffle=shuffle, dtype=dtype, random_state=random_state)
    elif dataset_name == 'fashion_mnist':
        shuffle = False
        random_state = 0
        (X_train, y_train), (X_test, y_test) = load_fashion_mnist(
            flatten=flatten, scale=scale, shuffle=shuffle, dtype=dtype, random_state=random_state)
    elif dataset_name == 'breast_cancer':
        shuffle = False
        random_state = 0
        scale = 'min_max_scaler'
        (X_train, y_train), (X_test, y_test) = load_breast_cancer(
            scale=scale, shuffle=shuffle, dtype=dtype, random_state=random_state)
    elif dataset_name == 'diabetes':
        scale = 'min_max_scaler'
        shuffle = False
        random_state = 0
        (X_train, y_train), (X_test, y_test) = load_diabetes(
            scale=scale, shuffle=shuffle, dtype=dtype, random_state=random_state, archive_path=archive_path)
    elif dataset_name == 'indian_liver_patient_records':
        scale = 'min_max_scaler'
        shuffle = False
        random_state = 0
        (X_train, y_train), (X_test, y_test) = load_indian_liver_patient(
            scale=scale, shuffle=shuffle, dtype=dtype, random_state=random_state, archive_path=archive_path)
    elif dataset_name == 'malaria_cell_images':
        scale = True
        shuffle = True
        random_state = 0
        (X_train, y_train), (X_test, y_test) = load_cell_images(
            scale=scale, shuffle=shuffle, dtype=dtype, random_state=random_state, archive_path=archive_path)

    print('Dataset {} loaded.'.format(dataset_name))
    print('{} training samples, {} testing samples.'.format(
        X_train.shape[0], X_test.shape[0]))
    return (X_train, y_train), (X_test, y_test)


def load_mnist(flatten=False, scale=True, shuffle=False, dtype='float32', random_state=0):

    (X_train, y_train_label), (X_test,
                               y_test_label) = tf.keras.datasets.mnist.load_data()
    y_train, y_test = tf.keras.utils.to_categorical(
        y_train_label), tf.keras.utils.to_categorical(y_test_label)

    n_train, n_test = X_train.shape[0], X_test.shape[0]
    X_train = X_train.astype(dtype)
    X_test = X_test.astype(dtype)

    if flatten == True:
        X_train, X_test = X_train.reshape(
            n_train, -1), X_test.reshape(n_test, -1)
    else:
        X_train, X_test = X_train.reshape(
            n_train, 28, 28, 1), X_test.reshape(n_test, 28, 28, 1)

    if scale == True:
        X_train, X_test = X_train / 255.0, X_test / 255.0

    if shuffle == True:
        X_train, y_train = shuffle_dataset(X_train, y_train, random_state)
        X_test, y_test = shuffle_dataset(X_test, y_test, random_state)

    return (X_train, y_train), (X_test, y_test)


def load_cifar10(flatten=False, scale=True, shuffle=False, dtype='float32', random_state=0):

    (X_train, y_train_label), (X_test,
                               y_test_label) = tf.keras.datasets.cifar10.load_data()
    y_train, y_test = tf.keras.utils.to_categorical(
        y_train_label), tf.keras.utils.to_categorical(y_test_label)

    n_train, n_test = X_train.shape[0], X_test.shape[0]

    X_train = X_train.astype(dtype)
    X_test = X_test.astype(dtype)

    if flatten == True:
        X_train, X_test = X_train.reshape(
            n_train, -1), X_test.reshape(n_test, -1)
    else:
        X_train, X_test = X_train.reshape(
            n_train, 32, 32, 3), X_test.reshape(n_test, 32, 32, 3)

    if scale == True:
        X_train, X_test = X_train / 255.0, X_test / 255.0

    if shuffle == True:
        X_train, y_train = shuffle_dataset(X_train, y_train, random_state)
        X_test, y_test = shuffle_dataset(X_test, y_test, random_state)

    return (X_train, y_train), (X_test, y_test)


def load_fashion_mnist(flatten=False, scale=True, shuffle=False, dtype='float32', random_state=0):

    (X_train, y_train_label), (X_test,
                               y_test_label) = tf.keras.datasets.fashion_mnist.load_data()
    y_train, y_test = tf.keras.utils.to_categorical(
        y_train_label), tf.keras.utils.to_categorical(y_test_label)

    n_train, n_test = X_train.shape[0], X_test.shape[0]
    X_train = X_train.astype(dtype)
    X_test = X_test.astype(dtype)

    if flatten == True:
        X_train, X_test = X_train.reshape(
            n_train, -1), X_test.reshape(n_test, -1)
    else:
        X_train, X_test = X_train.reshape(
            n_train, 28, 28, 1), X_test.reshape(n_test, 28, 28, 1)

    if scale == True:
        X_train, X_test = X_train / 255.0, X_test / 255.0

    if shuffle == True:
        X_train, y_train = shuffle_dataset(X_train, y_train, random_state)
        X_test, y_test = shuffle_dataset(X_test, y_test, random_state)

    return (X_train, y_train), (X_test, y_test)


def load_breast_cancer(scale=None, shuffle=False, dtype='float32', random_state=0):

    assert scale in {None, 'min_max_scaler',
                     'std_scaler'}, 'Parameter scale should be one of {None, \'min_max_scaler\', \'std_scaler\'} because the dataset is not a image dataset.'

    X, y = sklearn_dataset.load_breast_cancer(return_X_y=True)
    y = tf.keras.utils.to_categorical(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=shuffle, random_state=random_state)

    if scale is not None:
        if scale == 'min_max_scaler':
            scaler = MinMaxScaler(copy=False)
        elif scale == 'std_scaler':
            scaler = StandardScaler()
        scaler.fit_transform(X_train)
        scaler.transform(X_test)

    X_train = X_train.astype(dtype)
    X_test = X_test.astype(dtype)
    y_train = y_train.astype(dtype)
    y_test = y_test.astype(dtype)

    return (X_train, y_train), (X_test, y_test)


def load_diabetes(archive_path: Path, scale=None, shuffle=False, dtype='float32', random_state=0):

    assert scale in {None, 'min_max_scaler',
                     'std_scaler'}, 'Parameter scale should be one of {None, \'min_max_scaler\', \'std_scaler\'} because the dataset is not a image dataset.'

    if archive_path.is_file() == False:
        if archive_path.exists() == False:
            archive_path.mkdir()

        file_path = archive_path.joinpath('diabetes.csv')

        if file_path.exists() == False:
            raise 'Please download the dataset from https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database/, and move \'diabetes.csv\' into \'{}\.'.format(
                archive_path)

    else:
        file_path = archive_path

    df = pd.read_csv(file_path)

    X = df.drop('Outcome', axis=1).to_numpy()
    y = df['Outcome'].to_numpy()
    y = tf.keras.utils.to_categorical(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=shuffle, random_state=random_state)

    if scale is not None:
        if scale == 'min_max_scaler':
            scaler = MinMaxScaler(copy=False)
        elif scale == 'std_scaler':
            scaler = StandardScaler()
        scaler.fit_transform(X_train)
        scaler.transform(X_test)

    X_train = X_train.astype(dtype)
    X_test = X_test.astype(dtype)
    y_train = y_train.astype(dtype)
    y_test = y_test.astype(dtype)

    return (X_train, y_train), (X_test, y_test)


def load_indian_liver_patient(archive_path, scale=None, shuffle=False, dtype='float32', random_state=0):

    assert scale in {None, 'min_max_scaler',
                     'std_scaler'}, 'Parameter scale should be one of {None, \'min_max_scaler\', \'std_scaler\'} because this dataset is not a image dataset.'

    if archive_path.is_file() == False:
        if archive_path.exists() == False:
            archive_path.mkdir()
        file_path = archive_path.joinpath('indian_liver_patient.csv')

        if file_path.exists() == False:
            raise 'Please download the dataset from https://www.kaggle.com/datasets/uciml/indian-liver-patient-records, and move \'indian_liver_patient.csv\' into \'{}\'.'.format(
                str(archive_path))
    else:
        file_path = archive_path

    df = pd.read_csv(file_path).dropna()
    df['Gender'] = df['Gender'] == 'Male'
    df['Dataset'] -= 1

    X = df.drop('Dataset', axis=1).to_numpy()
    y = df['Dataset'].to_numpy()
    y = tf.keras.utils.to_categorical(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=shuffle, random_state=random_state)

    if scale is not None:
        if scale == 'min_max_scaler':
            scaler = MinMaxScaler(copy=False)
        elif scale == 'std_scaler':
            scaler = StandardScaler(copy=False)
        scaler.fit_transform(X_train)
        scaler.transform(X_test)

    X_train = X_train.astype(dtype)
    X_test = X_test.astype(dtype)
    y_train = y_train.astype(dtype)
    y_test = y_test.astype(dtype)

    return (X_train, y_train), (X_test, y_test)


def load_cell_images(archive_path, flatten=False, scale=True, shuffle=False, dtype='float32', random_state=0, image_shape=(32, 32, 3)):

    if archive_path.is_file() == False:
        if archive_path.exists() == False:
            archive_path.mkdir()
        npz_name = 'cell_images_({}_{}_{})_r_{}.npz'.format(
            image_shape[0], image_shape[1], image_shape[2], random_state)
        npz_path = archive_path.joinpath(npz_name)

        if npz_path.exists() == True:
            data = np.load(npz_path)
            X_train, y_train = data['X_train'], data['y_train']
            X_test, y_test = data['X_test'], data['y_test']
            return (X_train, y_train), (X_test, y_test)
            
        file_path = archive_path.joinpath('cell_images')
        if file_path.exists() == False:
            raise 'Please download the dataset from https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria, and move \'cell_images\' into \'{}\'.'.format(
                str(archive_path))
    else:
        file_path = archive_path

    # the path of positive samples and negative samples
    positive_path = file_path.joinpath('Parasitized')
    positive_list = os.listdir(positive_path)
    negative_path = file_path.joinpath('Uninfected')
    negative_list = os.listdir(negative_path)

    positive_list.remove('Thumbs.db')
    negative_list.remove('Thumbs.db')

    n_positive = len(os.listdir(positive_path))
    n_negative = len(os.listdir(negative_path))

    X = np.zeros(shape=(n_positive + n_negative, *image_shape))

    # positive first
    y = np.concatenate([np.ones(n_positive),
                       np.zeros(n_negative)])

    p = 0
    for cur_path, cur_list in zip((positive_path, negative_path), (positive_list, negative_list)):
        for i in range(len(cur_list)):
            img_path = cur_path.joinpath(cur_list[i])
            cur_img = Image.open(img_path)
            cur_img = cur_img.resize(image_shape[:2])
            X[p] = np.array(cur_img)
            p += 1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=shuffle, random_state=random_state)

    y_train, y_test = tf.keras.utils.to_categorical(
        y_train), tf.keras.utils.to_categorical(y_test)

    n_train, n_test = X_train.shape[0], X_test.shape[0]
    X_train = X_train.astype(dtype)
    X_test = X_test.astype(dtype)

    if flatten == True:
        X_train, X_test = X_train.reshape(
            n_train, -1), X_test.reshape(n_test, -1)

    if scale == True:
        X_train, X_test = X_train / 255.0, X_test / 255.0

    if shuffle == True:
        X_train, y_train = shuffle_dataset(X_train, y_train, random_state)
        X_test, y_test = shuffle_dataset(X_test, y_test, random_state)

    return (X_train, y_train), (X_test, y_test)


def second_to_str(second):
    second = int(second)
    hour = second // (60 * 60)
    second -= hour * 60 * 60
    minute = second // 60
    second -= minute * 60
    return '{:02}:{:02}:{:02}'.format(hour, minute, second)
