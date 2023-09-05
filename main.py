import os
import tensorflow as tf
from enum import Enum

from keras.src.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy

class DataType(Enum):
    CAT_DATA = ("PetImages/Cat",
                "cat_data.txt")
    DOG_DATA = ("PetImages/Dog",
                "dog_data.txt")


class DataInDirectoryNotFound(Exception):
    def __int__(self, message):
        self.message = message


def load_data(data_path):
    results = []
    for file_name in os.listdir(data_path):
        results.append(f"{data_path}/{file_name}")
    if results is None:
        raise DataInDirectoryNotFound
    return results


def validate_data(dpath):
    print("Validating %s" % dpath)
    data_object = open(dpath, "rb")
    is_valid = tf.compat.as_bytes("JFIF") in data_object.peek(10)
    data_object.close()
    return is_valid


def clean_data(data_paths):
    removed = 0
    for dpath in data_paths:
        is_valid = validate_data(dpath)
        if not is_valid:
            removed+=1
            data_paths.remove(dpath)
    print("Deleted %d images " % removed)


def save_data_to_file(data, fname):
    with open(f"{fname}", "w") as file:
        for item in data:
            file.write(f"{item}\n")


def load_data_from_file(fname):
    with open(fname, "r") as file:
        data = [line.strip() for line in file.readlines()]
    return data


def get_data(data_type):
    path = data_type.value[0]
    if not (os.path.exists(data_type.value[1]) and os.path.getsize(data_type.value[1]) > 0):
        data = load_data(path)
        clean_data(data)
        save_data_to_file(data, data_type.value[1])
        return data
    else:
        return load_data_from_file(data_type.value[1])



def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)

strategy = tf.distribute.MirroredStrategy()


if __name__ == '__main__':
    dog_data = get_data(DataType.DOG_DATA)
    cat_data = get_data(DataType.CAT_DATA)
    dog_data = dog_data[:(len(dog_data) // 2) - 1000]
    cat_data = cat_data[:(len(cat_data) // 2) - 1000]
    df = pd.DataFrame(
        {"obj_class": ["dog"] * len(dog_data) + ["cat"] * len(cat_data), "path": dog_data + cat_data})

    data_generator = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=0.2)

    training_generator = data_generator.flow_from_dataframe(
        dataframe=df,
        x_col='path',
        y_col='obj_class',
        target_size=(180,180),
        batch_size=32,
        class_mode='binary',
        subset='training',

    )
    validation_generator = data_generator.flow_from_dataframe(
        dataframe=df,
        x_col='path',
        y_col='obj_class',
        target_size=(180,180),
        batch_size=32,
        class_mode='binary',
        subset='validation'
    )
    '''
    for i in range(5):
        img, obj_class = next(generator)
        plt.imshow(img[0])
        plt.title(f'Class: {obj_class[0]}')
        plt.show()
    '''

    with strategy.scope():
        model = make_model(input_shape=(180, 180) + (3,), num_classes=2)
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

    callbacks = [
        keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
    ]

    # Trenowanie modelu wewnątrz strategii
    model.fit(
        training_generator,
        steps_per_epoch=len(training_generator),
        epochs=5,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        verbose=1,
        callbacks=callbacks
    )

    img = keras.utils.load_img(
        "PetImages/Dog/992.jpg", target_size=(180,180)
    )
    plt.imshow(img)
    img_array = keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    predictions = model.predict(img_array)
    score = float(predictions[0])
    print(f"This image is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog.")