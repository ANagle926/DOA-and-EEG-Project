import numpy as np
from PyEMD import EEMD
from joblib import load
from scipy.signal import welch
from scipy.stats import pearsonr
from itertools import product

from DataProcessingCNNV2 import DataProcessing
from VitalDBDataset import VitalDBDataset


def apply_AI_filter(x, b, c):
    data_processor = DataProcessing(x_eeg=x, y_doa=b, case_id=c)
    x=data_processor.cleaned_eeg
    b=data_processor.cleaned_y
    c=data_processor.cleaned_case_id
    return x, b, c

"""def grid_search_eemd(x_data, y_data, sample_size=100):
    x_data = x_data[:sample_size]
    y_data = y_data[:sample_size]

    best_config = None
    best_corr = -np.inf
    results = []

    max_imfs_list = [8]
    discard_first_n_list = [0, 1, 2]
    discard_last_n_list = [1, 2]
    noise_width_list = [0.03, 0.04]
    #max_imfs_list = [8]
    #discard_first_n_list = [2]
    #discard_last_n_list = [2]
    #noise_width_list = [0.03]

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

    return results, best_config"""

def extract_features(wave, fs=256):
    # Time-domain
    features = {}
    features['std'] = np.std(wave)
    features['ptp'] = np.ptp(wave)
    features['rms'] = np.sqrt(np.mean(wave**2))
    features['energy'] = np.sum(wave ** 2)

    # Frequency-domain
    f, Pxx = welch(wave, fs=fs)
    features['delta'] = np.sum(Pxx[(f >= 0.5) & (f < 4)])
    features['theta'] = np.sum(Pxx[(f >= 4) & (f < 8)])
    features['alpha'] = np.sum(Pxx[(f >= 8) & (f < 13)])
    features['beta'] = np.sum(Pxx[(f >= 13) & (f < 30)])
    features['gamma'] = np.sum(Pxx[(f >= 30)])

    return features

def grid_search_eemd(x_data, y_data, sample_size=100, fs=256):
    x_data = x_data[:sample_size]
    y_data = y_data[:sample_size]

    max_imfs_list = [8]
    discard_first_n_list = [0, 1, 2]
    discard_last_n_list = [0, 1, 2]
    noise_width_list = [0.03, 0.04]

    all_results = []
    feature_best_corr = {}  # e.g., {'std': (corr, config), ...}

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
            x_filtered = np.array([apply_custom_eemd(wave.squeeze()) for wave in x_data])
            all_features = [extract_features(wave, fs) for wave in x_filtered]

            feature_corrs = {}
            for key in all_features[0].keys():
                feature_vector = np.array([f[key] for f in all_features])
                corr = pearsonr(feature_vector, y_data.squeeze())[0]
                feature_corrs[key] = corr

                # Update best config per feature
                if key not in feature_best_corr or corr > feature_best_corr[key][0]:
                    feature_best_corr[key] = (corr, (max_imfs, discard_first, discard_last, noise_width))

            all_results.append((feature_corrs, (max_imfs, discard_first, discard_last, noise_width)))

            print(f"\n🧪 Config: IMFs={max_imfs}, Discard First={discard_first}, Last={discard_last}, Noise={noise_width}")
            for k, v in feature_corrs.items():
                print(f"→ Corr({k}): {v:.4f}")

        except Exception as e:
            print(f"❌ Failed for config: {max_imfs}, {discard_first}, {discard_last}, {noise_width}: {e}")

    print("\n🎯 Best Hyperparameters per Feature:")
    for feature, (corr, config) in feature_best_corr.items():
        print(f"{feature:>10} → Corr: {corr:.4f} | Config: IMFs={config[0]}, Discard First={config[1]}, Discard Last={config[2]}, Noise={config[3]}")

    return all_results, feature_best_corr

dataset = VitalDBDataset(max_cases=2, srate=128)
x=dataset.x
b=dataset.y
c=dataset.c
print("done loading dataset")

#x, b, c= apply_AI_filter(x, b, c)
grid_search_eemd(x,b)
