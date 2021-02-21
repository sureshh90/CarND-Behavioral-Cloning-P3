import csv
import cv2
import numpy as np
import math
import sklearn
samples = []

with open('/opt/carnd_p3/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    headers = next(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
        
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                correction = 0.2 
                for i in range(3):
                    name = '../../../opt/carnd_p3/data/IMG/'+batch_sample[i].split('/')[-1]
                    # Read the image and convert it to RGB color space
                    image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
                    images.append(image)
                    
                    center_angle = float(batch_sample[3])
                    if i == 0:
                        angle = center_angle
                        angles.append(angle)
                    elif i == 1:
                        angle = center_angle + correction
                        angles.append(angle)
                    elif i == 2:
                        angle = center_angle - correction
                        angles.append(angle)                    
                    
                    # Data augumentation: Add flipped images 
                    images.append(cv2.flip(image,1))
                    angles.append(angle*-1.0)

            # Add the images to the data set
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)        
        

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Activation
from keras.layers.convolutional import Conv2D
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


stopper = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=5)

model = Sequential()

# Preprocess the data using normalization and mean centering
model.add(Lambda(lambda x: (x / 255) - 0.55, input_shape=(160,320,3)))
# Crop the region of interest
model.add(Cropping2D(cropping=((70,25), (0,0))))

# The Neural Network Architecture from Nvidia
model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), padding='valid', activation='relu'))
model.add(Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), padding='valid', activation='relu'))
model.add(Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), padding='valid', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))

model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# Train the model
batch_size = 32
epochs = 15

# Compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

model.compile(loss='mse', optimizer='adam')

history_object = model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(train_samples)/batch_size), validation_data=validation_generator, validation_steps=math.ceil(len(validation_samples)/batch_size), epochs=epochs, callbacks=[stopper], verbose=1)

# Save the model file
model.save('model.h5')

### Plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('Model mean squared error loss')
plt.ylabel('Mean squared error loss')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper right')
plt.savefig('./writeup_images/03_all_cameras_stopper.jpg')
