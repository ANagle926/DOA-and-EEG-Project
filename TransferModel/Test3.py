import joblib
from keras import Sequential
from keras.src.optimizers import Adam
from keras.src.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.src.layers import MaxPooling1D, Bidirectional, LayerNormalization, LSTM, GlobalAveragePooling1D, Dense, \
    Dropout, Conv1D
import numpy as np
from typing import List, Tuple
from joblib import load
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import mean_absolute_error, r2_score
from EEGPT.downstream.Modules.models.EEGPT_mcae_finetune import EEGPTClassifier
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import load
import gc
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# === Utility ===
def get_device(force_cpu=False):
    if force_cpu or not torch.cuda.is_available():
        print("⚙️ Using CPU")
        return torch.device("cpu")
    print("⚡ Using GPU")
    return torch.device("cuda")


def forward_features_only(self, x):
    return self.forward_features(x, return_patch_tokens=True, return_all_tokens=False)

# === EEGPT Model Loader ===
def load_transfer_model(ckpt_path: str, channel_names: List[str]) -> torch.nn.Module:
    try:
        # First try loading to CUDA
        device = torch.device("cuda")
        checkpoint = torch.load(ckpt_path, map_location=device)
    except RuntimeError as e:
        print(f"⚠️ CUDA failed: {e}. Switching to CPU.")
        device = torch.device("cpu")
        checkpoint = torch.load(ckpt_path, map_location=device)

    model = EEGPTClassifier(
        eeg_size=(3, 1024),
        patch_size=64,
        emb_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        num_classes=2,
        use_channels_names=channel_names,
        desired_time_len=1024
    )
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.forward_features_only = forward_features_only.__get__(model)
    for param in model.parameters():
        param.requires_grad = False
    model.to(device)
    model.eval()
    return model


# === EEGPT Wrapper ===
class EEGPTWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model):
        self.model = model
        self.device = get_device()
        self.model.to(self.device)

    def fit(self, X, y=None, *args, **kwargs):
        return self

    def predict(self, X, *args, **kwargs):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X_tensor)
        return logits.argmax(dim=1).cpu().numpy()

    def predict_proba(self, X, batch_size=64, *args, **kwargs):
        self.model.eval()
        results = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(self.device)
                logits = self.model(batch)
                probs = F.softmax(logits, dim=1)
                results.append(probs.cpu())
        return torch.cat(results).numpy()


# === Feature Extraction ===
def extract_softmax_features(models: List[EEGPTWrapper],
                             x_data_list: List[np.ndarray],
                             batch_size: int = 64) -> np.ndarray:
    all_probs = [model.predict_proba(x, batch_size=batch_size) for model, x in zip(models, x_data_list)]
    stacked_probs = np.stack(all_probs, axis=1)
    return stacked_probs.reshape(stacked_probs.shape[0], -1)


def generate_voting_features(x_train: np.ndarray, x_test: np.ndarray,
                             y_train: np.ndarray, y_test: np.ndarray,
                             ckpt_path: str, channels: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    print("🧠 Extracting IMF-specific softmax features...")

    features_train, features_test = [], []

    for i in range(3):
        print(f" - Processing IMF {i+1}")

        # ✅ Clear memory BEFORE loading model
        gc.collect()
        torch.cuda.empty_cache()

        # Load model just for this IMF
        model = load_transfer_model(ckpt_path, channel_names=channels)
        wrapper = EEGPTWrapper(model)

        # Prepare IMF-specific input
        x_train_i = np.expand_dims(x_train[:, :, i], axis=1)
        x_test_i = np.expand_dims(x_test[:, :, i], axis=1)

        # Run inference in batches (GPU-safe)
        probs_train = wrapper.predict_proba(x_train_i, batch_size=64)
        probs_test = wrapper.predict_proba(x_test_i, batch_size=64)

        features_train.append(probs_train)
        features_test.append(probs_test)

        # ✅ Delete model after use
        del model, wrapper
        gc.collect()
        torch.cuda.empty_cache()

    # Stack all IMF features along feature axis
    x_train_out = np.concatenate(features_train, axis=1)
    x_test_out = np.concatenate(features_test, axis=1)

    print(f"✅ Extracted features: train={x_train_out.shape}, test={x_test_out.shape}")
    np.save("voted_train_features.npy", x_train_out)
    np.save("voted_test_features.npy", x_test_out)
    np.save("voted_train_labels.npy", y_train)
    np.save("voted_test_labels.npy", y_test)

    return x_train_out, x_test_out


# === BIS Regressor ===
def create_bis_regressor_model(x_train, y_train, x_test, y_test):
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    model = Sequential([
        Conv1D(64, kernel_size=2, activation='relu', input_shape=(x_train.shape[1], 1)),
        Dense(128, activation='relu', kernel_regularizer='l2'),
        GlobalAveragePooling1D(),
        Dropout(0.1),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(1e-4), loss='mse', metrics=['mae'])

    callbacks = [
        EarlyStopping(monitor='val_mae', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_mae', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    ]

    model.fit(x_train, y_train, validation_data=(x_test, y_test),
              epochs=80, batch_size=64, callbacks=callbacks, verbose=1)

    model.save("eeg_regressor.keras")
    return model

# === Evaluation ===
def evaluate_model(model, x_test, y_test):
    if x_test.ndim == 2:
        x_test = np.expand_dims(x_test, axis=-1)

    pred_test = model.predict(x_test).flatten()
    test_mae = mean_absolute_error(y_test, pred_test)
    corr = np.corrcoef(y_test, pred_test)[0, 1]
    r2 = r2_score(y_test, pred_test)

    print(f"\n\U0001f4ca Test MAE: {test_mae:.4f}")
    print(f"\U0001f4c8 Correlation coefficient: {corr:.4f}")
    print(f"\u2310 R² score: {r2:.4f}")

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, pred_test, s=1, alpha=0.5)
    plt.xlabel('Actual BIS')
    plt.ylabel('Predicted BIS')
    plt.title(f'Scatter Plot (Corr: {corr:.4f})')
    plt.plot([0, max(y_test)], [0, max(y_test)], 'r--')
    plt.grid(True)
    plt.show()

    errors = y_test - pred_test
    plt.figure(figsize=(8, 4))
    plt.hist(errors, bins=50, edgecolor='black')
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.title('Histogram of Prediction Errors')
    plt.grid(True)
    plt.show()

    abs_errors = np.abs(errors)
    plt.figure(figsize=(6, 6))
    sc = plt.scatter(y_test, pred_test, c=abs_errors, s=2, cmap='viridis', alpha=0.6)
    plt.xlabel('Actual BIS')
    plt.ylabel('Predicted BIS')
    plt.title('Colored Error Scatter Plot')
    plt.colorbar(sc, label='Absolute Error')
    plt.plot([0, max(y_test)], [0, max(y_test)], 'r--')
    plt.grid(True)
    plt.show()

def analyze_softmax_feature_correlation(x_softmax: np.ndarray, y_true: np.ndarray):
    """
    Analyze correlation between softmax outputs (6D) and true BIS values.
    Expects x_softmax shape = (samples, 6), and y_true shape = (samples,)
    """

    # Create labeled dataframe
    columns = ['IMF1_cls0', 'IMF1_cls1', 'IMF2_cls0', 'IMF2_cls1', 'IMF3_cls0', 'IMF3_cls1']
    df = pd.DataFrame(x_softmax, columns=columns)
    df['BIS'] = y_true

    # Compute correlation
    corrs = df.corr()['BIS'].drop('BIS')

    # Plot
    plt.figure(figsize=(8, 4))
    sns.barplot(x=corrs.index, y=corrs.values, palette="viridis")
    plt.title("Correlation of Softmax Outputs with BIS")
    plt.ylabel("Pearson Correlation")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return corrs  # return correlations if needed

from scipy.signal import periodogram

def analyze_dataset(x_raw, y_raw, fs=256, label="Train"):
    """
    Visual diagnostics for raw IMF EEG data and BIS labels.
    Expects x_raw shape = (samples, time, 3), where 3 = IMFs.
    """

    import matplotlib.pyplot as plt
    import numpy as np

    print(f"\n📊 Analyzing {label} dataset...")
    imf_labels = ['IMF 1', 'IMF 2', 'IMF 3']

    # === BIS Distribution
    plt.figure(figsize=(8, 4))
    plt.hist(y_raw, bins=50, color='mediumseagreen', edgecolor='black', alpha=0.7)
    plt.xlabel('BIS Value')
    plt.ylabel('Count')
    plt.title(f'{label} BIS Distribution')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # === Mean & Std by Timepoint
    mean = x_raw.mean(axis=0)  # shape (time, 3)
    std = x_raw.std(axis=0)

    for i in range(3):
        plt.figure(figsize=(12, 4))
        plt.plot(mean[:, i], label='Mean', color='navy')
        plt.plot(std[:, i], label='Std Dev', color='darkorange')
        plt.title(f'{label} IMF {i+1}: Mean & Std Across Time')
        plt.xlabel("Time Index")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # === Amplitude Histogram (avg abs per sample)
    amps = np.mean(np.abs(x_raw), axis=1)  # (samples, 3)
    for i in range(3):
        plt.figure(figsize=(8, 4))
        plt.hist(amps[:, i], bins=50, alpha=0.7, color='teal')
        plt.title(f'{label} IMF {i+1} Avg Amplitude')
        plt.xlabel('Avg |Amplitude|')
        plt.ylabel('Count')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # === Frequency Histograms (Dominant & Average)
    dom_freqs = np.zeros((x_raw.shape[0], 3))
    avg_freqs = np.zeros((x_raw.shape[0], 3))

    for i in range(x_raw.shape[0]):
        for imf in range(3):
            freqs, power = periodogram(x_raw[i, :, imf], fs=fs)
            dom_freqs[i, imf] = freqs[np.argmax(power)]
            avg_freqs[i, imf] = np.sum(freqs * power) / np.sum(power) if np.sum(power) > 0 else 0

    for i in range(3):
        plt.figure(figsize=(8, 4))
        plt.hist(dom_freqs[:, i], bins=50, alpha=0.6, label='Dominant Freq', color='slateblue')
        plt.hist(avg_freqs[:, i], bins=50, alpha=0.6, label='Avg Freq', color='darkcyan')
        plt.title(f'{label} IMF {i+1} Frequency Distribution')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# === Main Execution ===
dataset=load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/Data Files/dataset_twenty_cases_SEGLENMID.joblib")

x_train_raw, y_train = dataset.x_train, dataset.y_train
x_test_raw, y_test = dataset.x_test, dataset.y_test

ckpt_path = "/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/EEGPT/checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt"

channel_names = ['Fp1', 'Fp2', 'F3']

#analyze_dataset(x_train_raw, y_train, label="Train")
#analyze_dataset(x_test_raw, y_test, label="Test")

print("x_train_raw shape:", x_train_raw.shape)
print("x_test_raw shape:", x_test_raw.shape)

x_train, x_test = generate_voting_features(x_train_raw, x_test_raw, y_train, y_test,
                                           ckpt_path, channel_names)
model = create_bis_regressor_model(x_train, y_train, x_test, y_test)
evaluate_model(model, x_test, y_test)
analyze_softmax_feature_correlation(x_test, y_test)

