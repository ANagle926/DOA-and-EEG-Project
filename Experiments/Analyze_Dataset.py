from matplotlib import pyplot as plt
from scipy.signal import periodogram
import numpy as np
from VitalDBDataset import VitalDBDataset


def compute_dominant_frequencies(data, fs=256):
    """
    Computes dominant frequency per IMF channel for each sample.
    Returns: (samples, 3) array of dominant frequencies.
    """
    dominant_freqs = np.zeros((data.shape[0], 3))

    for i in range(data.shape[0]):
        for imf_idx in range(3):  # Assuming 3 IMFs
            freqs, power = periodogram(data[i, :, imf_idx], fs=fs)
            dominant_freq = freqs[np.argmax(power)]
            dominant_freqs[i, imf_idx] = dominant_freq

    return dominant_freqs

def compute_average_frequencies(data, fs=256):
    """
    Computes average frequency (spectral centroid) per IMF channel per sample.
    Returns: (samples, 3) array of weighted average frequencies.
    """
    avg_freqs = np.zeros((data.shape[0], 3))

    for i in range(data.shape[0]):
        for imf_idx in range(3):
            freqs, power = periodogram(data[i, :, imf_idx], fs=fs)
            total_power = np.sum(power)
            if total_power > 0:
                avg = np.sum(freqs * power) / total_power
            else:
                avg = 0
            avg_freqs[i, imf_idx] = avg

    return avg_freqs

def plot_frequency_histograms(x_train, x_test, fs=256):
    train_freqs_dominant = compute_dominant_frequencies(x_train, fs)
    test_freqs_dominant = compute_dominant_frequencies(x_test, fs)

    imf_labels = ['IMF 1', 'IMF 2', 'IMF 3']
    for i in range(3):
        plt.figure(figsize=(8, 4))
        plt.hist(train_freqs_dominant[:, i], bins=50, alpha=0.6, label='Train', color='skyblue')
        plt.hist(test_freqs_dominant[:, i], bins=50, alpha=0.6, label='Test', color='salmon')
        plt.title(f'Dominant Frequency Distribution - {imf_labels[i]}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    train_freqs_average = compute_average_frequencies(x_train, fs)
    test_freqs_average = compute_average_frequencies(x_test, fs)

    imf_labels = ['IMF 1', 'IMF 2', 'IMF 3']
    for i in range(3):
        plt.figure(figsize=(8, 4))
        plt.hist(train_freqs_average[:, i], bins=50, alpha=0.6, label='Train', color='skyblue')
        plt.hist(test_freqs_average[:, i], bins=50, alpha=0.6, label='Test', color='salmon')
        plt.title(f'Average Frequency Distribution - {imf_labels[i]}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def compute_average_amplitudes(data):
    """
    Computes average absolute amplitude per IMF channel per sample.
    Returns: (samples, 3) array of amplitudes.
    """
    return np.mean(np.abs(data), axis=1)

def plot_amplitude_histograms(x_train, x_test):
    train_amplitudes = compute_average_amplitudes(x_train)
    test_amplitudes = compute_average_amplitudes(x_test)

    imf_labels = ['IMF 1', 'IMF 2', 'IMF 3']
    for i in range(3):
        plt.figure(figsize=(8, 4))
        plt.hist(train_amplitudes[:, i], bins=50, alpha=0.6, label='Train', color='skyblue')
        plt.hist(test_amplitudes[:, i], bins=50, alpha=0.6, label='Test', color='salmon')
        plt.title(f'Average Amplitude Distribution - {imf_labels[i]}')
        plt.xlabel('Amplitude')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def analyze_dataset(x_train, y_train, x_test, y_test):

    plt.figure(figsize=(8, 4))
    plt.hist(y_train, bins=50, alpha=0.6, label='Train', color='skyblue')
    plt.hist(y_test, bins=50, alpha=0.6, label='Test', color='salmon')
    plt.xlabel('BIS Value')
    plt.ylabel('Count')
    plt.title('BIS Distribution: Train vs Test')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Shape: (samples, 64, 512)
    train_mean = x_train.mean(axis=(0, 1))  # mean of all tokens across samples
    test_mean = x_test.mean(axis=(0, 1))

    plt.figure(figsize=(12, 4))
    plt.plot(train_mean, label='Train Mean', color='blue')
    plt.plot(test_mean, label='Test Mean', color='red')
    plt.title("Mean Activation per Embedding Dimension")
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Mean Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    train_std = x_train.std(axis=(0, 1))
    test_std = x_test.std(axis=(0, 1))

    plt.figure(figsize=(12, 4))
    plt.plot(train_std, label='Train Std', color='blue')
    plt.plot(test_std, label='Test Std', color='red')
    plt.title("Std Deviation per Embedding Dimension")
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Standard Deviation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plot_amplitude_histograms(x_train, x_test)
    plot_frequency_histograms(x_train, x_test)


dataset=VitalDBDataset(max_cases=150)
x_test, y_test = dataset.x_test, dataset.y_test
x_train, y_train = dataset.x_train, dataset.y_train
analyze_dataset(x_train, y_train, x_test, y_test)