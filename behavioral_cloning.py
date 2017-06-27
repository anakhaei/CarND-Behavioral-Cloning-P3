import utils
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

images, measurements = utils.load_data('../data/')
X_train = np.array(images)
y_train = np.array(measurements)


model= utils.create_nvidia_model()

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model.h5')

#history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_epoch=5, verbose=1)

# print the keys contained in the history object
#print(history_object.history.keys())
# plot the training and validation loss for each epoch
#plt.plot(history_object.history['loss'])
#plt.plot(history_object.history['val_loss'])
#plt.title('model mean squared error loss')
#plt.ylabel('mean squared error loss')
#plt.xlabel('epoch')
#plt.legend(['training set', 'validation set'], loc='upper right')
#plt.show()

print(len(images))
print(len(measurements))
print(X_train[0].shape)
print(y_train[0].shape)
