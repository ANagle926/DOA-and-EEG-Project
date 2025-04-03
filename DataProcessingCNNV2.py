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
x_eeg = x_eeg.reshape((x_eeg.shape[0], x_eeg.shape[1], 1))  # Reshape for Conv1D input
data = x_eeg

std_threshold = 3
fft_threshold = 11000

def is_noisy(sample):
    std_dev = np.std(sample)
    fft_value = np.sum(np.abs(np.fft.fft(sample)))
    return (std_dev > std_threshold) or (fft_value > fft_threshold)

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
    plt.subplot(2, 2, 1)
    labels = ['Clean', 'Noisy']
    counts = [len(cleaned_data), len(noisy_indices)]
    plt.bar(labels, counts, color=['green', 'red'])
    plt.title('Distribution of Clean vs. Noisy Samples')
    plt.xlabel('Sample Type')
    plt.ylabel('Count')

    # Signal Amplitude Comparison
    plt.subplot(2, 2, 2)
    for i in range(3):
        plt.plot(cleaned_data[i], label=f'Clean {i+1}', color='green')
        plt.plot(data[noisy_indices[i]], label=f'Noisy {i+1}', color='red', linestyle='--')
    plt.title('Signal Amplitude: Clean vs. Noisy')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()

    # Signal Spectrum (FFT) Comparison
    plt.subplot(2, 2, 3)
    for i in range(3):
        fft_clean = np.abs(np.fft.fft(cleaned_data[i]))
        fft_noisy = np.abs(np.fft.fft(data[noisy_indices[i]]))
        plt.plot(fft_clean, label=f'Clean FFT {i+1}', color='green')
        plt.plot(fft_noisy, label=f'Noisy FFT {i+1}', color='red', linestyle='--')
    plt.title('Frequency Spectrum: Clean vs. Noisy')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.legend()

    # Heatmap of Noisy Signal Positions
    plt.subplot(2, 2, 4)
    heatmap = np.zeros(len(y))
    heatmap[noisy_indices] = 1
    plt.imshow([heatmap], cmap='hot', aspect='auto')
    plt.title('Heatmap of Noisy Signal Positions')
    plt.xlabel('Sample Index')
    plt.ylabel('Noise Presence')

    plt.tight_layout()
    plt.show()

#too many noisy samples for whatever reason => look at identify_noise, thats the correct amount (1/3)

#Prepare train/test data
labels = np.array([1 if is_noisy(sample) else 0 for sample in data])
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

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
print(f"Test Accuracy: {accuracy_score(y_test, y_pred)}")

#model= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/Data Files/Model Versions/data_processing.joblib")

cleaned_x, cleaned_y, noisy_indices = filter_noisy_data(x_eeg, y_doa, model)
visualize_noise_analysis(cleaned_x, noisy_indices, x_eeg, y_doa)

