import psutil

import numpy as np
from joblib import Parallel, delayed
import time
from PyEMD import EEMD

def apply_eemd_to_wave(wave, max_imfs=5):
    eemd = EEMD()
    imfs = eemd.eemd(wave)
    if imfs.shape[0] > max_imfs:
        imfs = imfs[:max_imfs]
    return np.sum(imfs, axis=0)

def process_dataset_eemd(x_data, max_imfs=5, n_jobs=2):
    print("⚙️ Starting parallel EEMD processing...")
    start_time = time.time()

    def process_single_wave(wave):
        return apply_eemd_to_wave(wave.squeeze(), max_imfs=max_imfs)

    processed = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(process_single_wave)(wave) for wave in x_data
    )

    total_time = time.time() - start_time
    print(f"✅ Done processing {len(x_data)} samples in {total_time:.2f} seconds.")
    return np.array(processed)[..., np.newaxis]

# Dummy EEG data: 10 samples, 512 points each
dummy_data = np.random.randn(10, 512, 1)
results = process_dataset_eemd(dummy_data)
print("Result shape:", results.shape)


"""dataset = Dataset2(max_cases=60, srate=128)

dump(dataset, "my_object.joblib")
#dataset = load("my_object.joblib")

x_train, y_train = dataset.x_train, dataset.y_train
x_test, y_test = dataset.x_test, dataset.y_test
c_train, c_test = dataset.c_train, dataset.c_test
print("c_test shape: ", c_test.shape)
print("y_test shape ",y_test.shape)
seglen = dataset.SEGLEN

# Initialize EEGRegressor object
eeg_regressor = EEGRegressor(x_train, y_train, x_test, y_test, c_test, seglen)
eeg_regressor.preprocess_data()
eeg_regressor.model = keras.models.load_model("eeg_regressor.keras")
eeg_regressor.plot_with_predictions()"""

"""from pympler import summary, muppy

def print_memory_usage():
    all_objects = muppy.get_objects()
    sum1 = summary.summarize(all_objects)
    summary.print_(sum1)

print_memory_usage()
print(f"Available memory: {psutil.virtual_memory().available / (1024 ** 3):.2f} GB")"""
