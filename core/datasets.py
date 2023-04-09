import pickle
import os

from PIL import Image
import math
import tensorflow as tf
import numpy as np
import random
from sklearn.model_selection import train_test_split


def open_f(filename, back=2):
    filepath = os.path.join('../' * back, 'data', filename)
    with open(filepath, 'rb') as f:
        return pickle.load(f)


class DataSequence(tf.keras.utils.Sequence):
    def __init__(self, x_set, y_set, transform=None, batch_size=512):
        self.x = x_set
        self.y = y_set
        self.transform = transform
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(self.y.shape[0] / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        shuffle = np.random.permutation(batch_x.shape[0])
        batch_x, batch_y = batch_x[shuffle], batch_y[shuffle]
        if 'torchvision.transforms' in str(type(self.transform)):
            return (
                np.array([np.asarray(self.transform(Image.fromarray(np.uint8(x)))) / 255. for x in batch_x]),
                batch_y
            )
        elif type(self.transform) is list:
            return (
                np.array([np.asarray(random.choice(self.transform)(Image.fromarray(np.uint8(x)))) / 255. for x in batch_x]),
                batch_y
            )
        return batch_x / 255, batch_y


def get_ds(file_train, file_test, transform=None, batch_size=512, one_hot=False, val_size=0.07, back=2):
    data_all = open_f(file_train, back)
    data_test = open_f(file_test, back)

    shuffle = np.random.permutation(data_all['images'].shape[0])
    train_images_full = data_all['images'][shuffle]
    train_labels_full = data_all['labels'][shuffle]
    if one_hot:
        train_labels_full = tf.one_hot(train_labels_full, 10).numpy()

    train_ds_x, val_ds_x, train_ds_y, val_ds_y = train_test_split(train_images_full, train_labels_full, test_size=val_size)

    val_ds = tf.data.Dataset.from_tensor_slices((val_ds_x / 255., val_ds_y))
    val_ds = val_ds.batch(batch_size)

    train_ds = DataSequence(train_ds_x, train_ds_y, transform, batch_size=batch_size)

    test_ds = data_test['images'] / 255.

    return train_ds, val_ds, test_ds
