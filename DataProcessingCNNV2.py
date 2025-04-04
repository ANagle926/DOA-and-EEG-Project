import numpy as np
from keras import Model, Sequential
from keras.src.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, BatchNormalization, ReLU
from keras.src.layers import Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import joblib
from joblib import dump, load

# Load data
y_doa = joblib.load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/Data Files/preprocess_b.joblib")
x_eeg = joblib.load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/Data Files/preprocess_x.joblib")

#8280
def is_noisy(sample, std_threshold=10, fft_threshold=12000, amplitude_change_threshold=0.3):
    # Statistical Method (Standard Deviation)
    std_dev = np.std(sample)  # Standard deviation of the sample

    # Frequency Domain Method (Fourier Transform)
    n = len(sample)  # Number of samples in the segment
    fft_values = np.abs(np.fft.fft(sample))  # Fourier Transform of the segment
    fft_values = fft_values[:n // 2]  # Only take the positive frequencies
    max_freq_component = np.max(fft_values)  # Maximum frequency component

    # Amplitude Change Detection
    # Calculate the absolute difference in amplitude between consecutive samples
    amplitude_changes = np.abs(np.diff(sample))
    mean_amplitude_change = np.mean(amplitude_changes)  # Average change in amplitude
    low_amplitude_change = mean_amplitude_change < amplitude_change_threshold

    # Check if the sample is noisy based on both criteria
    is_noisy_flag = (std_dev > std_threshold and max_freq_component > fft_threshold) or low_amplitude_change

    # Debugging information
    if is_noisy_flag:
        print(f"Detected noisy sample: std_dev={std_dev}, max_freq_component={max_freq_component}")

    return is_noisy_flag

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

#Filtering Data
def filter_noisy_data(data, y, model, threshold=0.5):
    # Predict noisy or clean using the model
    predictions = model.predict(data)
    predicted_labels = (predictions >= threshold).astype(int)

    cleaned_data = []
    cleaned_y = []
    noisy_indices = []

    for idx, label in enumerate(predicted_labels):
        if label == 0:  # Clean data
            cleaned_data.append(data[idx])
            cleaned_y.append(y[idx])
        else:
            noisy_indices.append(idx)

    cleaned_data = np.array(cleaned_data)
    cleaned_y = np.array(cleaned_y)

    print(f"Number of clean samples: {len(cleaned_data)}")
    print(f"Number of noisy samples: {len(noisy_indices)}")

    return cleaned_data, cleaned_y, noisy_indices

def visualize_noise_analysis(cleaned_data, noisy_indices, data, y):
    plt.figure(figsize=(16, 10))

    # Distribution of Noisy vs. Clean Samples
    """labels = ['Clean', 'Noisy']
    counts = [len(cleaned_data), len(noisy_indices)]
    plt.bar(labels, counts, color=['green', 'red'])
    plt.title('Distribution of Clean vs. Noisy Samples')
    plt.xlabel('Sample Type')
    plt.ylabel('Count')
    plt.show()

    plt.figure(figsize=(15, min(10, len([0, 1, 2]) * 3)))  # Adjust the figure size based on the number of comparisons
# Signal Amplitude Comparison
    for i in range(3):
        plt.plot(cleaned_data[i], label=f'Clean {i+1}', color='green')
        plt.plot(data[noisy_indices[i]], label=f'Noisy {i+1}', color='red')
    plt.title('Signal Amplitude: Clean vs. Noisy')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.tight_layout()
    plt.show()"""


    plt.figure(figsize=(15, min(10, len([0, 1, 2]) * 3)))  # Adjust the figure size based on the number of comparisons

    #Time Domain Comparison of Clean vs Noisy Signals
    for i, idx in enumerate([0, 100, 200, 300, 350]):  # You can modify the number of samples you want to compare
        plt.subplot(len([0, 1, 2, 3, 4]), 1, i + 1)

        # Clean signal
        plt.plot(cleaned_data[idx], label=f'Clean Signal {idx + 1}', color='green')
        # Noisy signal
        plt.plot(data[noisy_indices[idx]], label=f'Noisy Signal {idx + 1}', color='red')

        plt.title(f'Comparison of Clean vs Noisy Signal {idx + 1}')
        plt.xlabel('Time (Samples)')
        plt.ylabel('Amplitude')
        plt.legend()
    plt.tight_layout()
    plt.show()



#Prepare train/test data
"""labels = np.array([1 if is_noisy(sample) else 0 for sample in x_eeg])
noisy_count = np.sum(labels)
print(f"Total noisy samples detected: {noisy_count}")
X_train, X_test, y_train, y_test = train_test_split(x_eeg, labels, test_size=0.1, random_state=42)

# Apply SMOTE for class imbalance
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train.reshape(X_train.shape[0], -1), y_train)

# Reshape the training and testing sets for Conv1D
X_train_smote = X_train_smote.reshape(X_train_smote.shape[0], X_train_smote.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

#build and train model
model = create_model()
model.fit(X_train_smote, y_train_smote, epochs=10, validation_data=(X_test, y_test))
dump(model, "/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/Data Files/Model Versions/data_processing.joblib" )

#make predictions on testing set
y_pred = (model.predict(X_test) > 0.5).astype(int)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred)}")"""

model= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/Data Files/Model Versions/data_processing.joblib")

cleaned_x, cleaned_y, noisy_indices = filter_noisy_data(x_eeg, y_doa, model)
visualize_noise_analysis(cleaned_x, noisy_indices, x_eeg, y_doa)

