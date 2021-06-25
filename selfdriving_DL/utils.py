# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import cv2
import numpy as np
import os
import pandas as pd
import scipy.misc

from PIL import Image


def normalize_dataframe(data, num_bins=23):
    """
    Normalize dataframe to avoid bias to driving straight.

    :param data: Dataframe which is to be normalized.
    :param num_bins: Number of bins to use in angle histogram.

    :return: A Normalized dataframe.
    """
    avg_samples_per_bin = len(data['Steering Angle']) / num_bins
    hist, bins = np.histogram(data['Steering Angle'], num_bins)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) * 0.5

    # Drop random straight steering angles.
    keep_probs = []
    target = avg_samples_per_bin * .3
    for i in range(num_bins):
        if hist[i] < target:
            keep_probs.append(1.0)
        else:
            keep_probs.append(1.0 / (hist[i] / target))

    # Delete from X and y with probability 1 - keep_probs[j].
    remove_list = []
    for i in range(len(data['Steering Angle'])):
        angle = data['Steering Angle'][i]
        for j in range(num_bins):
            if angle > bins[j] and angle <= bins[j + 1]:
                if np.random.rand() > keep_probs[j]:
                    remove_list.append(i)

    data.drop(data.index[remove_list], inplace=True)
    return data


def load_data(file_name, columns):
    """
    Loads in data as dataframe ands sets column data type.

    :param file_name: Dataset file to read in.
    :param columns: Names of each column in dataset.

    :return: A dataframe.
    """
    data = pd.read_csv(file_name, names=columns, header=0)
    data[columns[:3]] = data[columns[:3]].astype(str)
    data[columns[3:]] = data[columns[3:]].astype(float)
    data = normalize_dataframe(data)
    images = data[columns[:3]]
    angles = data[columns[3]]
    return images, angles


def resize_crop(img):
    """
    Crops the image to focus only on road and then resizes it.

    :param img: Image to crop and resize.

    :return: A cropped image.
    """
    img = np.array(img, np.float32)
    img = img[60:140, :]
    #img = np.array(Image.fromarray(img).resize((200, 66)))
    #imgArray = Image.fromarray(np.uint8(img))
    #imgArray.save('a.jpg')
    return img


def jitter_image(path, steering):
    """
    Open image from disk and jitters it and modifies new angle.

    :param path: Path of image.
    :param steering: Steering angle corresponding to image.

    :return: Jittered image and new steering angle.
    """
    img = cv2.imread(path.strip())
    rows, cols, _ = img.shape
    transRange = 100
    numPixels = 10
    valPixels = 0.4
    transX = transRange * np.random.uniform() - transRange / 2
    transY = numPixels * np.random.uniform() - numPixels / 2
    transMat = np.float32([[1, 0, transX], [0, 1, transY]])
    steering = steering + transX / transRange * 2 * valPixels
    img = cv2.warpAffine(img, transMat, (cols, rows))
    return resize_crop(img), steering


def flip_image(path):
    """
    Flips the image.

    :param path: Path of image to flip.

    :return: A flipped image.
    """
    img = Image.open(path.strip())
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return resize_crop(img)


def tint_image(path):
    """
    Applies random tint to image to simulate night time.

    :param path: Path of image to flip.

    :return: A tinted image.
    """
    img = cv2.imread(path.strip())
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img = np.array(img, dtype=np.float64)
    random_bright = .5 + np.random.uniform()
    img[:, :, 2] = img[:, :, 2] * random_bright
    img[:, :, 2][img[:, :, 2] > 255] = 255
    img = np.array(img, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return resize_crop(img)


def load_image(path):
    """
    Loads an image give path.

    :param path: Path of image to flip.

    :return: An image.
    """
    img = Image.open(path)
    return resize_crop(img)
