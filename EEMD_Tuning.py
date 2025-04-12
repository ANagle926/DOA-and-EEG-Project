import numpy as np
from PyEMD import EEMD
from joblib import load
from scipy.stats import pearsonr
from itertools import product

from DataProcessingCNNV2 import DataProcessing


def apply_AI_filter(x, b, c):
    data_processor = DataProcessing(x_eeg=x, y_doa=b, case_id=c)
    x=data_processor.cleaned_eeg
    b=data_processor.cleaned_y
    c=data_processor.cleaned_case_id
    return x, b, c

def grid_search_eemd(x_data, y_data, sample_size=25):
    x_data = x_data[:sample_size]
    y_data = y_data[:sample_size]

    best_config = None
    best_corr = -np.inf
    results = []

    max_imfs_list = [8]
    discard_first_n_list = [0, 1, 2]
    discard_last_n_list = [1, 2]
    noise_width_list = [0.03, 0.035,  0.04]

    for max_imfs, discard_first, discard_last, noise_width in product(
            max_imfs_list, discard_first_n_list, discard_last_n_list, noise_width_list
    ):
        eemd = EEMD()
        eemd.noise_seed(42)
        eemd.noise_width = noise_width

        def apply_custom_eemd(wave):
            imfs = eemd.eemd(wave)
            if imfs.shape[0] > max_imfs:
                imfs = imfs[:max_imfs]
            kept_imfs = imfs[discard_first:imfs.shape[0] - discard_last]
            return np.sum(kept_imfs, axis=0) if kept_imfs.size > 0 else np.zeros_like(wave)

        try:
            x_filtered = np.array([
                apply_custom_eemd(wave.squeeze()) for wave in x_data
            ])
            mean_features = np.mean(x_filtered, axis=1)
            corr = pearsonr(mean_features, y_data.squeeze())[0]
        except Exception as e:
            print(f"❌ Failed for config: {max_imfs}, {discard_first}, {discard_last}, {noise_width}: {e}")
            corr = -np.inf

        results.append((corr, max_imfs, discard_first, discard_last, noise_width))

        if corr > best_corr:
            best_corr = corr
            best_config = (max_imfs, discard_first, discard_last, noise_width)

        print(f"→ Corr: {corr:.4f} | IMFs: {max_imfs}, Discard First: {discard_first}, Last: {discard_last}, Noise: {noise_width}")

    print("\n🎯 Best Configuration:")
    print(f"Corr: {best_corr:.4f}")
    print(f"IMFs: {best_config[0]}, Discard First: {best_config[1]}, Discard Last: {best_config[2]}, Noise: {best_config[3]}")

    return results, best_config

x=load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/Data Files/preprocess_x.joblib")
b=load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/Data Files/preprocess_b.joblib")
c=load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/Data Files/preprocess_c.joblib")

x, b, c= apply_AI_filter(x, b, c)
grid_search_eemd(x,b)
