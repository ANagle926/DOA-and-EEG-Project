import psutil

import numpy as np
from joblib import Parallel, delayed, load
import time
from PyEMD import EEMD
from matplotlib import pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score


"""old_dataset= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/Data Files/dataset_three_cases.joblib")
new_dataset= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/Data Files/dataset_twenty_cases.joblib")

# Shapes
print("📏 SHAPES:")
print("Old:", old_dataset.x_train.shape, old_dataset.y_train.shape)
print("New:", new_dataset.x_train.shape, new_dataset.y_train.shape)

# Value Ranges
print("\n🔢 TARGET VALUE RANGES:")
print("Old y: min =", np.min(old_dataset.y_train), "max =", np.max(old_dataset.y_train))
print("New y: min =", np.min(new_dataset.y_train), "max =", np.max(new_dataset.y_train))

# Target Distribution Comparison
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(old_dataset.y_train, bins=30, color='blue', kde=True)
plt.title("Old Dataset Target Distribution")

plt.subplot(1, 2, 2)
sns.histplot(new_dataset.y_train, bins=30, color='green', kde=True)
plt.title("New Dataset Target Distribution")
plt.tight_layout()
plt.show()

# EEG Value Ranges
print("\n📈 EEG VALUE RANGES:")
print("Old x: min =", np.min(old_dataset.x_train), "max =", np.max(old_dataset.x_train))
print("New x: min =", np.min(new_dataset.x_train), "max =", np.max(new_dataset.x_train))

# Plot a few EEG samples
def plot_eeg_segment(eeg_data, title):
    plt.figure(figsize=(10, 3))
    for i in range(eeg_data.shape[-1]):  # Plot each EEG channel
        plt.plot(eeg_data[:, i], label=f'Channel {i+1}')
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.show()

print("\n🧠 EEG SEGMENT EXAMPLES:")
plot_eeg_segment(old_dataset.x_train[2], "Old EEG Segment Example")
plot_eeg_segment(new_dataset.x_train[2], "New EEG Segment Example")"""

import tensorflow as tf
print(tf.__version__)


