import psutil

import numpy as np
from joblib import Parallel, delayed, load
import time
from PyEMD import EEMD
from scipy.stats import pearsonr


def calculate_correlation(x, y, name=""):
    print("contex is ", name)
    print(x.shape)
    print(y.shape)
    x_features = np.mean(x.squeeze(), axis=1)
    corr = pearsonr(x_features, y.squeeze())[0]
    print(f"Mean ➤ Correlation: {corr:.4f}")

x=load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/Data Files/preprocess_x.joblib")
y= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/Data Files/preprocess_b.joblib")
calculate_correlation(x,y, "EEMD")
