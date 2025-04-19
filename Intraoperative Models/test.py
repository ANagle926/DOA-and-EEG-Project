import psutil

import numpy as np
from joblib import Parallel, delayed, load
import time
from PyEMD import EEMD
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score

dataset=load("Processed_Data.joblib")
x_train, y_train = dataset.x_train, dataset.y_train
x_test, y_test = dataset.x_test, dataset.y_test
c_train, c_test = dataset.c_train, dataset.c_test
seglen = dataset.SEGLEN
model= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/Data Files/Model Versions/eeg_regressor.joblib")

# Predict and evaluate test statistics
pred_test = model.predict(x_test).flatten()

test_mae = mean_absolute_error(y_test, pred_test)
corr = np.corrcoef(y_test, pred_test)[0, 1]
r2 = r2_score(y_test, pred_test)
print(f"Test MAE: {test_mae:.4f}")
print(f"Correlation coefficient: {corr:.4f}")
print(f"R squared: {r2:.4f}")
