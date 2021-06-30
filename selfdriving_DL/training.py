# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import argparse
from PIL import Image
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy.misc
import copy

from IPython.display import display
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Input, Activation, Dense, Dropout, Flatten, Lambda, Conv2D, concatenate
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import csv

import utils # from utils.py


class DeepLearning():
    def __init__(self):
        kSEED = 5
        SIDE_STEERING_CONSTANT = 0.25
        NUM_BINS = 23
        NB_EPOCH = 30
        LR = 1e-3

        Center_Image = []
        Left_Image = []
        Right_Image = []
        Steering_Angle = []
        Throttle = []
        Break = []
        Speed = []

    def loaf_data(self):
        file_name = 'driving_log_2021.csv'

        with open(file_name) as f:

            reader = csv.reader(f)

            for row in reader:
                self.Center_Image.append(str(row[0].replace('/home/deepstation/Simulator/App/IMG_2021/', '')))
                self.Left_Image.append(str(row[1].replace('/home/deepstation/Simulator/App/IMG_2021/', '')))
                self.Right_Image.append(str(row[2].replace('/home/deepstation/Simulator/App/IMG_2021/', '')))
                self.Steering_Angle.append(str(row[3]))
                self.Throttle.append(str(row[4]))
                self.Break.append(str(row[5]))
                self.Speed.append(str(row[6]))
    
    def action_labels(self):
        angle_before = 0
        action = []
        l = len(self.Steering_Angle)
        #print(l)
        for i in range(l):
            angle = float(self.Steering_Angle[i]) - angle_before
            angle_before = float(self.Steering_Angle[i])
            action_kouho = [0, 0, 0, 0]
            #print(Steering_Angle[85],Throttle[85])
            if angle < 0 and float(self.Throttle[i]) >= 0:
                actionIndex = 0

            elif angle > 0 and float(self.Throttle[i]) >= 0:
                actionIndex = 1

            elif angle == 0 and float(self.Throttle[i]) > 0:
                actionIndex = 2
                
            elif angle == 0 and float(self.Throttle[i]) == 0:
                actionIndex = 3
                
            _action = copy.deepcopy(action_kouho)
            _action[actionIndex] = 1
            action.append(_action)
        return action

    def data_append(self, action):
        l = len(self.Center_Image)   
        data_1 = []
        data = []
        for i in range(l):
            data = np.append(data,self.Center_Image[i])
            data = np.append(data,self.Steering_Angle[i])
            data = np.append(data,self.Throttle[i])
            data_1.append(data)

        X_train, X_val, y_train, y_val = train_test_split(data_1, action, test_size=0.15, random_state=self.kSEED)
        return X_train, X_val, y_train, y_val

    def batch_generator(data, labels, augment_data=True, batch_size=64):

        batch_action = []
        batch_data = []
        batch_image = []
        sample_count = 0

        while True:
            # Shuffle indices to minimize overfitting.
            for i in np.random.permutation(len(data)):

                # Image (1) -> Center image and steering angle.
                center_path = data[i][0]
                angle = float(data[i][1])
                throttle = float(data[i][2])
                action = labels[i]

                center_image = utils.load_image(center_path)
                input_image = center_image
                input_data = [angle, throttle]

                batch_image.append(input_image)
                batch_data.append(input_data)        
                batch_action.append(action)
                #print(np.array(batch_data).shape)

                sample_count += 1

                if(augment_data):
                    if sample_count % 2 == 0:
                        center_image, angle = utils.jitter_image(center_path,angle)
                        input_image = center_image
                        input_data = [angle, throttle]

                        batch_image.append(input_image)
                        batch_data.append(input_data)        
                        batch_action.append(action)
                    else:
                        center_image= utils.tint_image(center_path)
                        input_image = center_image
                        input_data = [angle, throttle]

                        batch_image.append(input_image)
                        batch_data.append(input_data)        
                        batch_action.append(action)

                    sample_count += 1

                if ((sample_count % batch_size == 0) or (sample_count % len(data) == 0)):
                    #print(f"image.shape = {np.array(batch_image).shape}")
                    #print(f"data.shape = {np.array(batch_data).shape}")
                    yield [np.array(batch_image),np.array(batch_data)], np.array(batch_action)
                    # Reset batch/\n"
                    batch_image = []
                    batch_data = []
                    batch_action = []

    def data_array(self, X_train, y_train, X_val, y_val):
        generator_train = self.batch_generator(X_train, y_train, augment_data=True)
        generator_val = self.batch_generator(X_val, y_val, augment_data=False)
        return generator_train, generator_val

    def model(self, X_train, X_val, generator_train, generator_val):
        inputs_image= Input(shape=(80,320,3), name='image')
        inputs_data = Input(shape=(2,), name='data')
        conv2_1 = Conv2D(24, (5,5), padding='same', activation='relu', strides=(2,2))(inputs_image)
        maxp_1 = MaxPooling2D(pool_size=(2,2), strides=(1, 1))(conv2_1)
        conv2_2 = (Conv2D(36, (5,5), padding='same', activation='relu', strides=(2,2)))(maxp_1)
        maxp_2 = (MaxPooling2D(pool_size=(2,2), strides=(1, 1)))(conv2_2)
        conv2_3 = (Conv2D(48, (5,5), padding='same', activation='relu', strides=(2,2)))(maxp_2)
        maxp_3 = (MaxPooling2D(pool_size=(2,2), strides=(1, 1)))(conv2_3)
        conv2_4 = (Conv2D(64, (3,3), padding='same', activation='relu', strides=(1,1)))(maxp_3)
        maxp_4 = (MaxPooling2D(pool_size=(2,2), strides=(1, 1)))(conv2_4)
        conv2_5 = (Conv2D(64, (3,3), padding='same', activation='relu', strides=(1,1)))(maxp_4)
        maxp_5 = (MaxPooling2D(pool_size=(2,2), strides=(1, 1)))(conv2_5)
        flat = Flatten()(maxp_5)
        action = Dense(4, activation="softmax", name='action')(flat)
        x = concatenate([action, inputs_data])
        common1 = Dense(64, input_dim = 2, activation="relu", name='common1')(x)
        common2 = Dense(64, activation="relu", name='common2')(common1)
        action_fin = Dense(4, activation="softmax", name='action_fin')(common2)
        model = keras.Model(inputs=[inputs_image,inputs_data], outputs=action_fin)
        model.compile(optimizer=Adam(lr=LR, decay=LR / NB_EPOCH), loss='mse', metrics=['accuracy'])

        model.fit_generator(
                generator_train,
                steps_per_epoch=2 * len(X_train),
                epochs=self.NB_EPOCH,
                validation_data=generator_val,
                validation_steps=len(X_val),
                verbose=1)

if __name__ == '__main__':
    deeplearning = DeepLearning()
