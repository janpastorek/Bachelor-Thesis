import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import layers
from keras.models import Sequential
from keras.models import model_from_json
from keras.optimizers import Adam

from RegressionModel import RegressionModel


def override(f): return f


def show_history(history, block=True):
    fig, axs = plt.subplots(2, 1, num='Training history', sharex=True)

    plt.subplot(2, 1, 1)
    plt.title('Regression error per epoch')
    plt.plot(history.history['loss'], '-b', label='training loss')
    try:
        plt.plot(history.history['val_loss'], '-r', label='validation loss')
    except KeyError:
        pass
    plt.grid(True)
    plt.legend(loc='best')
    plt.xlim(left=-1);
    plt.ylim(bottom=-0.01)

    plt.subplot(2, 1, 2)
    plt.title('Classification accuracy per epoch [%]')
    plt.plot(np.array(history.history['accuracy']) * 100, '-b', label='training accuracy')
    try:
        plt.plot(np.array(history.history['val_acc']) * 100, '-r', label='validation accuracy')
    except KeyError:
        pass
    plt.grid(True)
    plt.legend(loc='best')
    plt.xlim(left=-1);
    plt.ylim(-3, 103)

    plt.tight_layout()
    plt.show(block=block)


class KerasModel(RegressionModel):
    """ Regression model using more layers """

    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    def __init__(self, input_dim, n_action):
        # Build the model
        self.dnn = Sequential()

        self.dnn.add(layers.Dense(input_dim, activation='relu', input_shape=[input_dim]))
        self.dnn.add(layers.Dense(32, activation='relu'))

        # output layer
        self.dnn.add(layers.Dense(n_action))

        self.compiled = False
        self.losses = None

    @override
    def predict(self, X):
        return self.dnn.predict(X)

    @override
    def sgd(self, X, Y, learning_rate, momentum):
        # Train the model

        if not self.compiled:
            self.dnn.compile(loss='mse',
                             optimizer=Adam(lr=learning_rate, beta_1=momentum, beta_2=0.999),
                             metrics=['mae'])
            self.compiled = True

        history = self.dnn.fit(X, Y,
                               epochs=1,
                               batch_size=1,
                               verbose=3
                               )

        # show_history(history)

    @override
    def load_weights(self, _):
        # load json and create model
        json_file = open('../.training/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.dnn = model_from_json(loaded_model_json)
        # load weights into new model
        self.dnn.load_weights(".training/model.h5")
        print("Loaded model from disk")

    @override
    def save_weights(self, _):
        # serialize model to JSON
        model_json = self.dnn.to_json()
        with open("../.training/model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.dnn.save_weights(".training/model.h5")
        print("Saved model to disk")

    @override
    def get_losses(self):
        return self.losses
