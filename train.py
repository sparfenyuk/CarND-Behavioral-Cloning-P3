import csv
import cv2
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data", type=str, default='./data',
	help="(optional) folder where data images are located")
ap.add_argument("-l", "--load-model", type=int, default=-1,
	help="(optional) whether or not pre-trained model should be loaded")
ap.add_argument("-i", "--offset", type=int, default=1,
	help="(optional) read images with offset, convenient for debugging")
ap.add_argument("-e", "--epochs", type=int, default=2,
	help="(optional) read images with offset, convenient for debugging")
args = vars(ap.parse_args())

DATA_FOLDER = args['data']
lines = []
with open(DATA_FOLDER + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []

print("Preparing data...")

offset = args["offset"] # for DEBUG purposes

"""Load images from DATA_FOLDER"""
for line in lines[::offset]:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = DATA_FOLDER + '/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

"""Augment images: flip them horizontally"""
def run_augmentation():
    global images, measurements
    aug_images = []
    aug_measurements = []
    for image, measurement in zip(images, measurements):
        flipped_image = cv2.flip(image, 0)
        flipped_measurement = -measurement
        aug_images.append(flipped_image)
        aug_measurements.append(flipped_measurement)
    images += aug_images
    measurements += aug_measurements

run_augmentation()

"""Setup training data"""
X_train = np.array(images)
y_train = np.array(measurements)
EPOCHS = args["epochs"]

print("Done. {} images loaded".format(X_train.shape[0]))

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

"""Create our model"""
"""LeNet architecture"""
model = Sequential()
model.add(Lambda(lambda x: x/255. - .5, input_shape=(160, 320, 3)))
model.add(Convolution2D(6,5,5, activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.5))
model.add(Convolution2D(6,5,5, activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(120, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=.2, shuffle=True, nb_epoch=EPOCHS)

model.save('model.h5')
