'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization, GlobalAveragePooling2D, Rescaling
from keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import EarlyStopping


batch_size = 512
num_classes = 10
epochs = 10

# Data augmentation and normalization layers
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])
normalization_layer = tf.keras.layers.Rescaling(1./255)

with tf.device('/cpu:0'):
    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape for CNN input (batch, height, width, channels)
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32')
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32')
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # Early stopping callback
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    # Improved CNN model architecture
    model = Sequential([
        data_augmentation,
        normalization_layer,
        Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        BatchNormalization(),
        MaxPooling2D(),
        Conv2D(64, 3, activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),
        Conv2D(128, 3, activation='relu'),
        MaxPooling2D(),
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])

    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=15,
        verbose=1,
        validation_data=(x_test, y_test),
        callbacks=[early_stop]
    )
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
