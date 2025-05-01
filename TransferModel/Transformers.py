from tsai.all import *
import torch
import numpy as np
import numpy as np
from keras import layers, models


# Load pretrained InceptionTime model
model = load_learner('InceptionTime_Inference_Learner.pkl')  # You may need to download one

# Prepare your IMF EEG input
# X shape: (num_samples, seq_len, n_channels) e.g., (5000, 125, 3)
X = np.load('imf_data.npy')  # Your IMF segments

# Convert to torch format
X_tensor = torch.from_numpy(X).float()

# Disable gradients
model.model.eval()
with torch.no_grad():
    # Pass data through all but the final layer to get feature vectors
    features = model.model[0](X_tensor)  # assume model[0] is the CNN encoder

# Save features for Keras
np.save("inception_features.npy", features.numpy())


# Load saved features
X_features = np.load("inception_features.npy")  # shape: (samples, feature_dim)
y_labels = np.load("bis_labels.npy")  # Your BIS values (same order as X)

# Keras regression model
inputs = layers.Input(shape=(X_features.shape[1],))
x = layers.Dense(128, activation='relu')(inputs)
x = layers.Dropout(0.3)(x)
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(1, activation='linear')(x)

model = models.Model(inputs, outputs)
model.compile(optimizer='adam', loss='mae', metrics=['mae'])

# Train model
model.fit(X_features, y_labels, validation_split=0.2, epochs=20, batch_size=64)
