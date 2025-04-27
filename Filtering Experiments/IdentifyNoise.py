from itertools import product

import numpy as np
from joblib import load
from matplotlib import pyplot as plt
from numpy.fft import fft, fftfreq
from scipy.stats import pearsonr
from tqdm import tqdm

#Number of clean samples: 56037
#Number of noisy samples: 8820

x=load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/Data Files/x_data_without_filter.joblib")
y=load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/Data Files/y_data_without_filter.joblib")
print(x.shape)
print(y.shape)

# Choose a few random segments
#indices = np.random.choice(len(x), 3, replace=False)  # Random indices for 5 segments

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

#43687 noisy

def visualize_noisy_segments(data, noisy_segments):
    num_noisy = len(noisy_segments)
    if num_noisy == 0:
        print("No noisy segments detected.")
        return

    # Plot the noisy segments
    plt.figure(figsize=(15, min(10, num_noisy * 3)))  # Adjust the figure size based on the number of noisy segments
    for i, idx in enumerate(noisy_segments[:5]):  # Limit to first 5 noisy segments for visualization
        plt.subplot(min(5, num_noisy), 1, i + 1)
        plt.plot(data[idx])
        plt.title(f"EEG Segment {idx} (Noisy)")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.show()

def calculate_correlation(x, y):
    x_features = np.mean(x.squeeze(), axis=1)
    corr = pearsonr(x_features, y.squeeze())[0]
    return corr

#fft threshold 20-22
def identify_noisy_segments(data, std_threshold=14, fft_threshold=9000, amplitude_change_threshold=0.2):
    noisy_segments = []

    for idx, segment in enumerate(data):
        std_dev = np.std(segment)
        n = len(segment)
        fft_values = np.abs(fft(segment))[:n // 2]
        max_freq_component = np.max(fft_values)
        amplitude_changes = np.abs(np.diff(segment))
        mean_amplitude_change = np.mean(amplitude_changes)
        low_amplitude_change = mean_amplitude_change < amplitude_change_threshold
# "and" or "or"
        if (std_dev > std_threshold and max_freq_component > fft_threshold) or low_amplitude_change:
            noisy_segments.append(idx)

    return noisy_segments

def grid_search_noise_thresholds(x_data, y_data, std_range, fft_range, amplitude_change_threshold=0.2):
    best_corr = -np.inf
    best_params = (None, None)

    # Materialize all combinations
    all_combinations = list(product(std_range, fft_range))
    print("🔍 Starting grid search over thresholds...")

    for i, (std_thresh, fft_thresh) in enumerate(tqdm(all_combinations, total=len(all_combinations))):
        print(f"\n🔧 Combo {i+1}/{len(all_combinations)} → STD: {std_thresh}, FFT: {fft_thresh}")

        noisy_indices = identify_noisy_segments(x_data.squeeze(), std_thresh, fft_thresh, amplitude_change_threshold)
        mask = np.ones(len(x_data), dtype=bool)
        mask[noisy_indices] = False

        x_clean = x_data[mask]
        y_clean = y_data[mask]

        if len(x_clean) == 0 or len(y_clean) == 0:
            print("⚠️ Skipping this combo: all data was filtered out.")
            continue

        x_features = np.mean(x_clean.squeeze(), axis=1)
        corr = pearsonr(x_features, y_clean.squeeze())[0]
        print(f"📊 Correlation = {corr:.4f}")

        if corr > best_corr:
            best_corr = corr
            best_params = (std_thresh, fft_thresh)
            print(f"🌟 New best! STD={std_thresh}, FFT={fft_thresh}, Corr={corr:.4f}")

    print("\n✅ Finished grid search.")
    print(f"🏆 Best STD = {best_params[0]}, FFT = {best_params[1]} ➤ Corr = {best_corr:.4f}")
    return best_params, best_corr


"""std_range = np.arange(13, 16, 1) #13-15
fft_range = np.arange(8000,11000, 1000) #8000-9000

best_params, best_corr = grid_search_noise_thresholds(x, y, std_range, fft_range)"""

noisy_segments = identify_noisy_segments(x)
print(f"Number of noisy segments detected: {len(noisy_segments)}")
visualize_noisy_segments(x, noisy_segments)
