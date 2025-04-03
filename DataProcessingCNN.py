import numpy as np
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from joblib import load
from keras import Model, Sequential
from keras.src.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, BatchNormalization, ReLU
from keras.src.layers import Flatten, Dense
from keras.src.optimizers import Adam
from matplotlib import pyplot as plt
from numpy.fft import fft, fftfreq
from sklearn.model_selection import train_test_split


x=load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/Data Files/preprocess_x.joblib")
x = x.reshape((*x.shape, 1))
print(x.shape)
data=x

# Step 1: Data Classification
std_threshold = 3
fft_threshold = 11000

def is_noisy(sample):
    std_dev = np.std(sample)
    fft_value = np.sum(np.abs(np.fft.fft(sample)))
    return (std_dev > std_threshold) or (fft_value > fft_threshold)

# Assuming 'data' is your EEG dataset of shape (num_samples, 512)
labels = np.array([1 if is_noisy(sample) else 0 for sample in data])

# Step 2: Dataset Preparation (without SMOTE)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=42)

# Flatten data for SMOTE
X_train_flat = X_train.reshape((X_train.shape[0], -1))

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_flat, y_train)

unique, counts = np.unique(y_train_resampled, return_counts=True)
print("Number of samples after SMOTE:")
for label, count in zip(unique, counts):
    label_name = "Noisy" if label == 1 else "Clean"
    print(f"{label_name} samples: {count}")

# Reshape back to (samples, 512, 1) after SMOTE
X_train_resampled = X_train_resampled.reshape((-1, 512, 1))


# Step 3: Binary Classification CNN Model
def create_model():
    model = Sequential([
        Input(shape=(512, 1)),
        Conv1D(32, 3, activation='relu'),
        MaxPooling1D(2),
        Conv1D(64, 3, activation='relu'),
        MaxPooling1D(2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = create_model()

# Step 4: Model Training
history = model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_test, y_test))

# Step 5: Visual Analysis
def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

plot_history(history)

loss, accuracy = model.evaluate(X_test, y_test)

print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")



