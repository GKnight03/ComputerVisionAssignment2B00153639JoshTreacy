


import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Dropout, Flatten, Rescaling, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report



batch_size = 32
num_classes = 3
epochs = 10
img_width = 128
img_height = 128
img_channels = 3
fit = True  # Set to False to load a pre-trained model


# Data augmentation and normalization layers
data_augmentation = Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
])
normalization_layer = Rescaling(1. / 255)

train_dir = r'C:\Users\fires\OneDrive\Desktop\ComputerVisions\Assignment2B00153639JoshTreacy\chest_xray\chest_xray\train'
test_dir = r'C:\Users\fires\OneDrive\Desktop\ComputerVisions\Assignment2B00153639JoshTreacy\chest_xray\chest_xray\test'


# Create training, validation, and test datasets
train_ds, val_ds = tf.keras.preprocessing.image_dataset_from_directory(
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
print('Class Names:', class_names)
num_classes = len(class_names)

# Visualize some training images
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(6):
        ax = plt.subplot(2, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i].numpy()])
        plt.axis("off")
plt.show()

# MobileNetV2 transfer learning model
base_model = MobileNetV2(
    input_shape=(img_height, img_width, img_channels),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # freeze it

model = Sequential([
    Input(shape=(img_height, img_width, img_channels)),
    data_augmentation,
    normalization_layer,
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Early stopping and model checkpoint callbacks
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)
save_callback = ModelCheckpoint("pneumonia.keras", save_freq='epoch', save_best_only=True)

# Class weights to handle imbalance
class_weight = {
    0: 1.0,  # BACTERIAL
    1: 1.5,  # NORMAL (less aggressive)
    2: 1.2   # VIRAL
}

# Train or load model
if fit:
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[early_stop, save_callback],
        class_weight=class_weight
    )
else:
    model = tf.keras.models.load_model("pneumonia.keras")

# Evaluate model on test dataset
score = model.evaluate(test_ds)
print("Test accuracy:", score[1])

# Plot training history if trained
if fit:
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

# Classification report on test set
y_true = []
y_pred = []
for images, labels in test_ds:
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Visualize predictions on a test batch
test_batch = test_ds.take(1)
plt.figure(figsize=(10, 10))
for images, labels in test_batch:
    for i in range(6):
        ax = plt.subplot(2, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        prediction = model.predict(tf.expand_dims(images[i].numpy(), 0))
        plt.title('Actual: ' + class_names[labels[i].numpy()] +
                  '\nPredicted: {} {:.2f}%'.format(class_names[np.argmax(prediction)], 100 * np.max(prediction)))
        plt.axis("off")
plt.show()
