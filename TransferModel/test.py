import os
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.4/targets/x86_64-linux/lib"

import tensorflow as tf
from keras.src.layers import MaxPooling1D, Bidirectional, LayerNormalization, LSTM, GlobalAveragePooling1D, Dense
from keras import Model, Input
import numpy as np


print("TensorFlow version:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices('GPU'))

# Generate dummy input: batch_size=8, timesteps=10, features=16
x_input = np.random.randn(8, 10, 16).astype(np.float32)

# Define the model
inputs = Input(shape=(10, 16))
outputs = Bidirectional(LSTM(32))(inputs)
model = Model(inputs, outputs)
model.summary()

# Forward pass using __call__, not .predict()
y_output = model(x_input)
print("Output shape:", y_output.shape)