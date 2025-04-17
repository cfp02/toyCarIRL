"""
The design of this comes from here:
http://outlace.com/Reinforcement-Learning-Part-3/
"""

from keras.callbacks import Callback  # type: ignore
from keras.layers import LSTM, Activation, Dense, Dropout  # type: ignore
from keras.models import Sequential  # type: ignore
from keras.optimizers import RMSprop  # type: ignore


class LossHistory(Callback):
    def on_train_begin(self, logs=None):
        self.losses = []

    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        self.losses.append(logs.get("loss"))


def neural_net(num_sensors, params, load=""):
    model = Sequential()

    # First layer
    model.add(
        Dense(params[0], kernel_initializer="lecun_uniform", input_shape=(num_sensors,))
    )
    model.add(Activation("relu"))
    model.add(Dropout(0.2))

    # Second layer
    model.add(Dense(params[1], kernel_initializer="lecun_uniform"))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))

    # Output layer
    model.add(Dense(3, kernel_initializer="lecun_uniform"))
    model.add(Activation("linear"))

    rms = RMSprop(learning_rate=0.001)
    model.compile(loss="mse", optimizer=rms)

    if load:
        model.load_weights(load)

    return model


def lstm_net(num_sensors, load=False):
    model = Sequential()
    model.add(LSTM(units=512, input_shape=(None, num_sensors), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=512, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(3))
    model.add(Activation("linear"))

    model.compile(loss="mean_squared_error", optimizer=RMSprop(learning_rate=0.001))

    if load:
        model.load_weights(load)

    return model
