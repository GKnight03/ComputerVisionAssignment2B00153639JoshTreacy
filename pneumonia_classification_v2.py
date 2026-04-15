

from __future__ import print_function

import keras
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Rescaling, BatchNormalization, GlobalAveragePooling2D
from keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np


batch_size = 12
num_classes = 3
epochs = 8
img_width = 128
img_height = 128
img_channels = 3
fit = True #make fit false if you do not want to train the network again

# Data augmentation and normalization layers
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_dir = r'C:\Users\fires\OneDrive\Desktop\ComputerVisions\Assignment2B00153639JoshTreacy\chest_xray\chest_xray\train'
test_dir = r'C:\Users\fires\OneDrive\Desktop\ComputerVisions\Assignment2B00153639JoshTreacy\chest_xray\chest_xray\test'

with tf.device('/gpu:0'):
    
    #create training,validation and test datatsets
    train_ds,val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        seed=123,
        validation_split=0.2,
        subset='both',
        image_size=(img_height, img_width),
        batch_size=batch_size,
        labels='inferred',
        shuffle=True)
    
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        seed=None,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        labels='inferred',
        shuffle=True)

    class_names = train_ds.class_names
    print('Class Names: ',class_names)
    num_classes = len(class_names)
    
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(2):
        for i in range(6):
            ax = plt.subplot(2, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i].numpy()])
            plt.axis("off")
    plt.show()

    # Improved model architecture with data augmentation, normalization, batchnorm, and GAP
    model = tf.keras.Sequential([
        data_augmentation,
        normalization_layer,
        Conv2D(32, 3, activation='relu', input_shape=(img_height, img_width, img_channels)),
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

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])

    # Early stopping callback
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    save_callback = tf.keras.callbacks.ModelCheckpoint("pneumonia.keras", save_freq='epoch', save_best_only=True)

    # Class weights to handle imbalance
    class_weight = {
        from tensorflow.keras.callbacks import EarlyStopping
        0: 1.0,  # BACTERIAL
        1: 2.0,  # NORMAL (boost this)
        2: 1.5   # VIRAL
    }

        epochs = 15
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=15,
            class_weight=class_weight,
            callbacks=[early_stop, save_callback]
        )
    else:
        model = tf.keras.models.load_model("pneumonia.keras")

    score = model.evaluate(test_ds, batch_size=batch_size)
    print('Test accuracy:', score[1])

    if fit:
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

    test_batch = test_ds.take(1)
    plt.figure(figsize=(10, 10))
    for images, labels in test_batch:
        for i in range(6):
            ax = plt.subplot(2, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            prediction = model.predict(tf.expand_dims(images[i].numpy(), 0))
            plt.title('Actual:' + class_names[labels[i].numpy()] + '\nPredicted:{} {:.2f}%'.format(class_names[np.argmax(prediction)], 100 * np.max(prediction)))
            plt.axis("off")
    plt.show()
