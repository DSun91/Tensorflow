import os
import zipfile
import random
import tensorflow as tf
import shutil
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil
from os import getcwd

path_cats_and_dogs = "cats-and-dogs.zip"

local_zip = path_cats_and_dogs
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall()
zip_ref.close()

try:
    base_dir=os.mkdir('cats-v-dogs/')
    train=os.mkdir('cats-v-dogs/training/')
    test=os.mkdir('cats-v-dogs/testing/')
    train_d=os.mkdir('cats-v-dogs/training/dogs/')
    train_c=os.mkdir('cats-v-dogs/training/cats/')
    test_d=os.mkdir('cats-v-dogs/testing/dogs/')
    test_c=os.mkdir('cats-v-dogs/testing/cats/')
except OSError:
    print(OSError)
    pass


def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    complete_list = os.listdir(SOURCE)

    split_train_size = round(len(complete_list) * SPLIT_SIZE)
    split_test_size = round(len(complete_list) * (1 - SPLIT_SIZE))

    print(split_train_size, split_test_size)

    training_list = random.sample(complete_list, split_train_size)
    testing_list = random.sample(complete_list, split_test_size)

    for i in range(0, split_train_size):
        shutil.copy(f'{SOURCE}%s' % training_list[i], f'{TRAINING}%s' % training_list[i])

    for i in range(0, split_test_size):
        shutil.copy(f'{SOURCE}%s' % testing_list[i], f'{TESTING}%s' % testing_list[i])


CAT_SOURCE_DIR = "PetImages/Cat/"
TRAINING_CATS_DIR = "cats-v-dogs/training/cats/"
TESTING_CATS_DIR = "cats-v-dogs/testing/cats/"
DOG_SOURCE_DIR = "PetImages/Dog/"
TRAINING_DOGS_DIR = "cats-v-dogs/training/dogs/"
TESTING_DOGS_DIR = "cats-v-dogs/testing/dogs/"

split_size = .9
split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(16, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')#tf.keras.layers.Dense(n_classes, activation='softmax') and y= tf.keras.getcategorical o similar  
])                                                #loss Categorical_crossentropy

model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])

size=150
train_datagen = ImageDataGenerator(rescale=1/255,
                                   rotation_range=60,
                                   width_shift_range=0.3,
                                   height_shift_range=0.3,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')#YOUR CODE HERE

# NOTE: YOU MUST USE A BATCH SIZE OF 10 (batch_size=10) FOR THE
# TRAIN GENERATOR.
train_generator = train_datagen.flow_from_directory(
    'cats-v-dogs/training/',
    target_size=(size,size),
    batch_size=10,
    class_mode='binary',
)#YOUR CODE HERE

validation_datagen = ImageDataGenerator(rescale=1/255,
                                   rotation_range=60,
                                   width_shift_range=0.3,
                                   height_shift_range=0.3,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')#YOUR CODE HERE

# NOTE: YOU MUST USE A BACTH SIZE OF 10 (batch_size=10) FOR THE
# VALIDATION GENERATOR.
validation_generator = validation_datagen.flow_from_directory(
    'cats-v-dogs/testing/',
    target_size=(size,size),
    batch_size=10,
    class_mode='binary')#YOUR CODE HERE

history = model.fit_generator(train_generator,
                              epochs=4,
                              verbose=1,
                              validation_data=validation_generator)

model.save("my_model")


import matplotlib.image  as mpimg
import matplotlib.pyplot as plt

#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")


plt.title('Training and validation loss')

# Desired output. Charts with training and validation metrics. No crash :)