#if were using CNN instead of YOLO11 then we can use the following code, rn its image classifcation
#difference is that YOLO11 uses "bounding boxes" to detect objects so its a little bit harder

import tensorflow as tf
from import layers, models
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def load_data():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0
    return (train_images, train_labels), (test_images, test_labels)

def build_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10))  # from_logits=True if we keep the loss as is

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model

def train_model(model, train_images, train_labels, test_images, test_labels, epochs=10, batch_size=64):
    
    datagen = ImageDataGenerator(# Data augmentation
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    datagen.fit(train_images)

    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min')

    history = model.fit(
        datagen.flow(train_images, train_labels, batch_size=batch_size),
        epochs=epochs,
        steps_per_epoch=len(train_images) // batch_size,
        validation_data=(test_images, test_labels),
        callbacks=[early_stop, checkpoint]
    )
    return history

def evaluate_model(model, test_images, test_labels):
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f'\nTest accuracy: {test_acc:.4f}')

def visualize_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Train')
    plt.plot(epochs_range, val_acc, label='Val')
    plt.legend()
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Train')
    plt.plot(epochs_range, val_loss, label='Val')
    plt.legend()
    plt.title('Loss')
    plt.show()

def visualize_predictions(model, test_images, test_labels):
    class_names = ['airplane','automobile','bird','cat','deer',
                   'dog','frog','horse','ship','truck']

    predictions = model.predict(test_images)
    num_examples = 5
    indices = np.random.choice(range(len(test_images)), num_examples, replace=False)

    plt.figure(figsize=(10, 10))
    for i, idx in enumerate(indices):
        ax = plt.subplot(1, num_examples, i + 1)
        plt.imshow(test_images[idx])
        pred_label = np.argmax(predictions[idx])
        true_label = test_labels[idx][0]
        color = 'green' if pred_label == true_label else 'red'
        plt.title(f'Pred: {class_names[pred_label]}\nTrue: {class_names[true_label]}', color=color)
        plt.axis('off')
    plt.show()

if __name__ == "__main__":
    (train_images, train_labels), (test_images, test_labels) = load_data()
    model = build_model()
    history = train_model(model, train_images, train_labels, test_images, test_labels, epochs=20)
    visualize_training_history(history)
    evaluate_model(model, test_images, test_labels)
    visualize_predictions(model, test_images, test_labels)

