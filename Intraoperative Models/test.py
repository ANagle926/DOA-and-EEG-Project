import psutil

import numpy as np
from joblib import Parallel, delayed, load
import time
from PyEMD import EEMD
from matplotlib import pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score


import os
import os

# Verify dataset path
import os

# Absolute path to dataset
data_dir = os.path.expanduser("~/sleep-edf-expanded")  # /home/anika/sleep-edf-expanded

# Verify in code
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Dataset not found at: {data_dir}")

print("Dataset contents:", os.listdir(data_dir))


required_folders = ["sleep-cassette", "sleep-telemetry"]

for folder in required_folders:
    path = os.path.join(data_dir, folder)
    if not os.path.exists(path):
        print(f"❌ Missing folder: {path}")
    else:
        print(f"✅ Found: {path} ({len(os.listdir(path))} files)")



