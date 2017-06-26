import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt


lines = []
with open ('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements= []
correction = 0.3
for line in lines:
    source_path_center = line[0]
    source_path_left = line[1]
    source_path_right = line[2]
    filename_center = source_path_center.split('/')[-1]
    filename_left = source_path_left.split('/')[-1]
    filename_right = source_path_right.split('/')[-1]
    #current_path = '../data/IMG/' + filename
    image_center = cv2.imread('../data/IMG/' + filename_center)
    image_left = cv2.imread('../data/IMG/' + filename_left)
    image_right = cv2.imread('../data/IMG/' + filename_right)
    
    measurement = float(line[3])
    images.append (image_center)
    measurements.append(measurement)
    images.append (image_left)
    measurements.append(measurement+correction)
    images.append (image_right)
    measurements.append(measurement-correction)

    images.append (np.fliplr(image_center))
    measurements.append(-1 * measurement)

    images.append (np.fliplr(image_left))
    measurements.append(-1 * (measurement+correction))

    images.append (np.fliplr(image_right))
    measurements.append(-1 * (measurement-correction))

lines = []
with open ('../run-1/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

for line in lines:
    source_path_center = line[0]
    source_path_left = line[1]
    source_path_right = line[2]
    filename_center = source_path_center.split('/')[-1]
    filename_left = source_path_left.split('/')[-1]
    filename_right = source_path_right.split('/')[-1]

    image_center = cv2.imread('../run-1/IMG/' + filename_center)
    image_left = cv2.imread('../run-1/IMG/' + filename_left)
    image_right = cv2.imread('../run-1/IMG/' + filename_right)
    
    measurement = float(line[3])
    images.append (image_center)
    measurements.append(measurement)
    images.append (image_left)
    measurements.append(measurement+correction)
    images.append (image_right)
    measurements.append(measurement-correction)

    images.append (np.fliplr(image_center))
    measurements.append(-1 * measurement)

    images.append (np.fliplr(image_left))
    measurements.append(-1 * (measurement+correction))

    images.append (np.fliplr(image_right))
    measurements.append(-1 * (measurement-correction))


X_train=np.array(images)
y_train=np.array(measurements) 

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Convolution2D(6, 5, 5, activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6, 5, 5, activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6, 5, 5, activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(1024))
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model.h5')

################################################################ 
# Save the model and weights 
################################################################ 
model_json = model.to_json() 
with open("./model_2.json", "w") as json_file: 
    json.dump(model_json, json_file) 
model.save_weights("./model_2.h5") 
print("Saved model to disk")
##############################################################

history_object = model.fit_generator(train_generator, samples_per_epoch =
    len(train_samples), validation_data = 
    validation_generator,
    nb_val_samples = len(validation_samples), 
    nb_epoch=5, verbose=1)

### print the keys contained in the history object
print(history_object.history.keys())
### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

print (len(images))
print (len(measurements))
print (X_train[0].shape)
print (y_train[0].shape)