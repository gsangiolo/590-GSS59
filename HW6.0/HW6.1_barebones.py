# BAREBONES AUTOENCODER IMPLEMENTATION BASED ON THE ARTICLE HERE: https://blog.keras.io/building-autoencoders-in-keras.html

from keras import layers, Input, Model
from keras.datasets import mnist, fashion_mnist
import numpy as np
import matplotlib.pyplot as plt

# Dataset options: mnist, fashion_mnist
def load_data(dataset='mnist'):
    if dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    elif dataset == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    else:
        return None

    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    input_shape = x_train.shape[1] * x_train.shape[2]

    x_train = x_train.reshape(x_train.shape[0], input_shape)
    x_test = x_test.reshape(x_test.shape[0], input_shape)
    return x_train, x_test


def build_autoencoder(input_shape, bottleneck):
    input_layer = Input(shape=(input_shape,))

    encoding_layer = layers.Dense(bottleneck, activation='relu')(input_layer)

    decoding_layer = layers.Dense(input_shape, activation='sigmoid')(encoding_layer)

    autoencoder = Model(input_layer, decoding_layer)

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return autoencoder

x_train, x_test = load_data()

autoencoder = build_autoencoder(x_train.shape[1], 32)

history = autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

fashion_train, fashion_test = load_data('fashion_mnist')

autoencoder.predict(fashion_test)
