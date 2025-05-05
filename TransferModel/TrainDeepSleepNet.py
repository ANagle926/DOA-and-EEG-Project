from tsai.all import *
dsid = 'ECG5000'

# Get data with splits
X, y, splits = get_UCR_data(dsid, split_data=False)
X_train = X[splits[0]]  # Get training portion
y_train = y[splits[0]]  # Get training labels

# Define transforms with EXPLICIT vocabulary
tfms = [None, TSCategorize()]
batch_tfms = TSStandardize()

# Create dataloaders
dls = get_ts_dls(
    X, y, tfms=tfms, splits=splits, batch_tfms=batch_tfms
)

# Create learner
learn = ts_learner(dls, arch='InceptionTime', metrics=accuracy)
learn.fit_one_cycle(20, 1e-3)
learn.export("InceptionTime_ECG5000.pkl")

import numpy as np
import torch

# Load trained model
learn = load_learner("InceptionTime_ECG5000.pkl")
learn.model.eval()

# Load your IMF EEG data
# Shape should be: (samples, channels=3, timesteps=125)
X_imf = np.load("your_imf_data.npy")
X_tensor = torch.from_numpy(X_imf).float()

# Extract features using the CNN encoder only
with torch.no_grad():
    features = learn.model[0](X_tensor)  # CNN feature extractor

# Save the extracted features for Keras
np.save("extracted_features.npy", features.numpy())

import numpy as np
from tensorflow.keras import layers, models

# Load features and corresponding BIS values
X_features = np.load("extracted_features.npy")
y_bis = np.load("bis_labels.npy")  # shape: (samples,)

# Build regression model
inputs = layers.Input(shape=(X_features.shape[1],))
x = layers.Dense(128, activation='relu')(inputs)
x = layers.Dropout(0.3)(x)
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(1, activation='linear')(x)

model = models.Model(inputs, outputs)
model.compile(optimizer='adam', loss='mae', metrics=['mae'])

# Train
model.fit(X_features, y_bis, validation_split=0.2, epochs=20, batch_size=64)
