import csv
import cv2
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data", type=str, default='./data',
	help="(optional) folder where data images are located")
ap.add_argument("-l", "--load-model", type=int, default=-1,
	help="(optional) whether or not pre-trained model should be loaded")
ap.add_argument("-a", "--augment", type=int, default=1,
	help="(optional) whether or not augment images: 0,1")
ap.add_argument("-e", "--epochs", type=int, default=2,
	help="(optional) read images with offset, convenient for debugging")
args = vars(ap.parse_args())

DATA_FOLDER = args['data']

def load_logs():
    global DATA_FOLDER
    with open(DATA_FOLDER + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        return [line for line in reader]

CROP_TOP = 70
CROP_BOTTOM = 30

def load_image(source_path):
    image = cv2.imread(source_path)
    if image is None:
        raise Exception("Probably you've got broken dataset")
    return image

STEERING_CORRECTION = 0.4

"""Load images from DATA_FOLDER"""
def load_data(lines, batch_size, use_augmentation):
    shuffle(lines)
    size = len(lines)
    while 1:
        correction = STEERING_CORRECTION
        for start_i in range(0, size, batch_size):
            images, measurements = [],[]
            for line in lines[start_i:start_i+batch_size]:
                for i in range(3):
                    image = load_image(line[i])
                    images.append(image)
                measurement = float(line[3])
                measurements += [measurement,
                                 measurement + STEERING_CORRECTION,
                                 measurement - STEERING_CORRECTION];
            if use_augmentation:
                run_augmentation(images, measurements)
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield shuffle(X_train, y_train)

"""Augment images: flip them horizontally"""
def run_augmentation(images, measurements):
    aug_images = []
    aug_measurements = []
    for image, measurement in zip(images, measurements):
        flipped_image = cv2.flip(image, 0)
        flipped_measurement = -measurement
        aug_images.append(flipped_image)
        aug_measurements.append(flipped_measurement)
    images += aug_images
    measurements += aug_measurements


print("Preparing data...")
lines = load_logs()
print('There are {} records in config'.format(len(lines)))
augment_images = bool(args['augment'])
train_samples, validation_samples = train_test_split(lines, test_size=0.2)
train_generator = load_data(train_samples, batch_size=32, use_augmentation=augment_images)
validation_generator = load_data(validation_samples, batch_size=32, use_augmentation=augment_images)


"""Setup training data"""
EPOCHS = args["epochs"]

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import EarlyStopping, TensorBoard
"""Create our model"""
model = Sequential()
"""Nvidia"""
model.add(Lambda(lambda x: x/255. - .5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((CROP_TOP, CROP_BOTTOM), (0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(26,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,1,1,activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')

coef = (2 if augment_images else 1) * 3 # 3 stands for three front cameras
model.fit_generator(train_generator,
                    samples_per_epoch=len(train_samples) * coef,
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples) * coef,
                    nb_epoch=EPOCHS, callbacks=[
                                        EarlyStopping(min_delta=0.001),
                                        TensorBoard(),
                                      ])

import os
import datetime

model_name = "model-{}.h5".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
model.save(model_name)
print('Model saved to ' + model_name)
symlink_name = 'model.h5'
try:
    os.remove(symlink_name)
except OSError:
    pass
os.symlink(model_name, symlink_name)
