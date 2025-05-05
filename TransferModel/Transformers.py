import numpy as np
from fastai.learner import load_learner
from fastai.metrics import accuracy
from keras import layers, models
from tsai.data.core import TSClassification
from tsai.data.external import get_UCR_data
from tsai.data.preprocessing import TSStandardize
from tsai.tslearner import TSClassifier

# Load ECG5000 dataset
X, y, splits = get_UCR_data('ECG5000', split_data=True)

# Define transformations
tfms = [None, TSClassification()]
batch_tfms = TSStandardize()

# Initialize and train the model
clf = TSClassifier(X, y, splits=splits, path='models', arch="InceptionTime", tfms=tfms, batch_tfms=batch_tfms, metrics=accuracy)
clf.fit_one_cycle(20, 1e-3)

# Export the trained model
clf.export("InceptionTime_ECG5000.pkl")

# Load the trained model
learn = load_learner("InceptionTime_ECG5000.pkl")

# Prepare your EEG IMF data
# Ensure X_imf has the shape: (samples, variables, timesteps)
X_imf = np.load("your_imf_data.npy")

# Extract features
features = learn.get_X_preds(X_imf, with_input=False)
