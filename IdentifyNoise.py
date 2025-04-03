import numpy as np
from joblib import load
from matplotlib import pyplot as plt
from numpy.fft import fft, fftfreq

x=load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/Data Files/preprocess_x.joblib")
print(x.shape)

# Choose a few random segments
indices = np.random.choice(len(x), 3, replace=False)  # Random indices for 5 segments

"""plt.figure(figsize=(15, 10))
for i, idx in enumerate(indices):
    plt.subplot(5, 1, i + 1)
    plt.plot(x[idx])
    plt.title(f"EEG Segment {idx}")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()

# Distribution of values for a random segment
plt.hist(x[10], bins=50)
plt.title(f"Distribution of EEG Segment {10}")
plt.xlabel("Amplitude")
plt.ylabel("Frequency")
plt.show()"""

# Function to identify noisy segments using both statistical and frequency domain methods
def identify_noisy_segments(data, std_threshold=3, fft_threshold=11000):
    # List to store indices of noisy segments
    noisy_segments = []

    # Iterate through each EEG segment
    for idx, segment in enumerate(data):
        # Statistical Method (Standard Deviation)
        std_dev = np.std(segment)  # Standard deviation of the segment

        # Frequency Domain Method (Fourier Transform)
        n = len(segment)  # Number of samples in the segment
        frequencies = fftfreq(n)  # Frequency bins
        fft_values = np.abs(fft(segment))  # Fourier Transform of the segment
        fft_values = fft_values[:n // 2]  # Only take the positive frequencies
        max_freq_component = np.max(fft_values)  # Maximum frequency component

        # Check if segment is noisy based on both criteria
        if std_dev > std_threshold and max_freq_component > fft_threshold:
            noisy_segments.append(idx)

    return noisy_segments

def visualize_noisy_segments(data, noisy_segments):
    num_noisy = len(noisy_segments)
    if num_noisy == 0:
        print("No noisy segments detected.")
        return

    # Plot the noisy segments
    plt.figure(figsize=(15, min(10, num_noisy * 3)))  # Adjust the figure size based on the number of noisy segments
    for i, idx in enumerate(noisy_segments[:5]):  # Limit to first 5 noisy segments for visualization
        plt.subplot(min(5, num_noisy), 1, i + 1)  # Adjust the subplot grid to fit noisy segments
        plt.plot(data[idx])
        plt.title(f"EEG Segment {idx} (Noisy)")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.show()

# Identify noisy segments based on both statistical and frequency domain methods
noisy_segments = identify_noisy_segments(x)

# Output the result
print(f"Number of noisy segments detected: {len(noisy_segments)}")

# Visualize the noisy segments
visualize_noisy_segments(x, noisy_segments)