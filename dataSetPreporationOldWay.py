import os 
import glob
import random
import pandas as pd
from PIL import Image
import numpy as np


def get_data_set(path):

    # Loading the location of all files - image dataset
    # Considering our image dataset has apple or orange
    # The images are named as apple01.jpg, apple02.jpg .. , orange01.jpg .. etc.
    images_path_list = glob.glob(path+'*.png')
    images = []

    for image in images_path_list:
        img = Image.open(image)
        img = np.array(img.convert('L').getdata())
        img = img / 255.0
        images.append(img)

    # print(images)

    # Shuffling the dataset to remove the bias - if present
    random.shuffle(images)

    #reading a csv
    df1 = pd.read_csv('tData/tData.csv')
    # print(df1)
    one_hot = pd.get_dummies(df1['c_type'])
    df1 = df1.drop('c_type', axis=1)
    df1 = df1.join(one_hot)
    # print(df1)

    labels = df1.iloc[:, 1:].values #geting values of a DataFrame column to a numpy array
    # print(labels)

    print(type(images))

    split_size = int(0.6 * len(images))

    # Splitting the dataset
    training_images = images[:split_size]
    training_labels = labels[:split_size]
    testing_images  = images[split_size:]
    testing_labels  = labels[split_size:]

    # print(training_images)
    return training_images, training_labels, testing_images, testing_labels

# get_data_set("tData/TEST/")