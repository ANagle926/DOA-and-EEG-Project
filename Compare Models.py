import numpy as np
import pandas as pd
import psutil
from joblib import load
from keras import Input, Model, Sequential
from keras.src.layers import Conv1D, MaxPooling1D, Bidirectional, GlobalAveragePooling1D, LSTM, \
    MultiHeadAttention, LayerNormalization, Dropout, Dense
from keras import layers
from keras.src.saving import register_keras_serializable
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.svm import SVR
import tensorflow as tf


"""def extract_band_features(fft_mag, freqs):
    
    Extracts 5 stats:
    - Beta band (13–30 Hz): std, min, entropy
    - Theta band (4–8 Hz): median, rms
    
    beta_idx = np.where((freqs >= 13) & (freqs <= 30))[0]
    theta_idx = np.where((freqs >= 4) & (freqs <= 8))[0]

    beta = fft_mag[beta_idx]
    theta = fft_mag[theta_idx]

    features = []

    # Beta band
    features.append(np.std(beta))
    features.append(np.min(beta))
    if beta.sum() == 0:
        features.append(0)
    else:
        features.append(entropy(beta / beta.sum(), base=2))

    # Theta band
    features.append(np.median(theta))
    features.append(np.sqrt(np.mean(theta ** 2)))  # RMS

    return features  # length = 5

def wfa_feature_extractor(x_data, srate=128, wavelet='db6', level=5):
    
    Applies DWT → select D3 → FFT → extract band-based stats
    Returns: (N, 5) feature matrix
    
    if x_data.ndim == 2:
        x_data = x_data[:, np.newaxis, :]  # shape (N, 1, T)

    all_features = []

    for i in range(x_data.shape[0]):
        signal = x_data[i, 0]

        coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level)
        d3 = coeffs[-3]  # D3 from [A5, D5, D4, D3, D2, D1]

        fft_mag = np.abs(np.fft.rfft(d3))
        freqs = np.fft.rfftfreq(len(d3), d=1/srate)

        feats = extract_band_features(fft_mag, freqs)
        all_features.append(feats)

    return np.array(all_features)  # shape (N, 5)

def FSE_feature_extraction(x_data, fs=128, band=(21.5, 38.5)):
    
    Apply Wavelet Fourier Analysis style preprocessing from the paper:
    - FFT on each EEG segment
    - Extract power in beta-gamma band (21.5–38.5 Hz)
    - Compute 10 statistical features:
      min, entropy, mean, std, RMS, var, skew, kurtosis, range, mode
    
    if x_data.ndim == 2:
        x_data = x_data[:, np.newaxis, :]  # (N, 1, T)

    n_samples, n_channels, n_timesteps = x_data.shape
    features = []

    for i in range(n_samples):
        sample_feats = []
        for j in range(n_channels):
            signal = x_data[i, j]
            freqs = rfftfreq(n_timesteps, d=1/fs)
            fft_vals = np.abs(rfft(signal))
            band_mask = (freqs >= band[0]) & (freqs <= band[1])
            band_vals = fft_vals[band_mask]

            if len(band_vals) == 0:
                band_vals = np.zeros(10)

            psd = band_vals ** 2
            psd_sum = np.sum(psd)
            if psd_sum == 0:
                psd_norm = np.ones_like(psd) / len(psd)
            else:
                psd_norm = psd / psd_sum

            # 10 features
            sample_feats.extend([
                np.min(psd),                               # XMin
                entropy(psd_norm, base=2),                 # Xentropy
                np.mean(psd),                              # Mean
                np.std(psd),                               # XsD
                np.sqrt(np.mean(psd ** 2)),                # XrMs
                np.var(psd),                               # XVar
                skew(psd),                                 # Xske
                kurtosis(psd),                             # XKurt
                np.ptp(psd),                               # XRang
                mode(psd, keepdims=False)[0]               # XMod
            ])
        features.append(sample_feats)

    features = np.array(features)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize
    features -= features.mean(axis=0)
    features /= features.std(axis=0) + 1e-6
    return features.astype(np.float32)"""

@register_keras_serializable()
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length

    def build(self, input_shape):
        d_model = input_shape[-1]
        self.token_proj = layers.Dense(d_model)
        self.position_embeddings = layers.Embedding(input_dim=self.sequence_length, output_dim=d_model)

    def call(self, x):
        length = tf.shape(x)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        pos_encoding = self.position_embeddings(positions)
        return x + pos_encoding

    def get_config(self):
        config = super().get_config()
        config.update({"sequence_length": self.sequence_length})
        return config
@register_keras_serializable()
class TransformerBlock(layers.Layer):
    def __init__(self, num_heads, key_dim, ff_units, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.ff_units = ff_units
        self.dropout_rate = dropout_rate

        self.attn = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.attn_norm = LayerNormalization()
        self.ffn_norm = LayerNormalization()

    def build(self, input_shape):
        embed_dim = input_shape[-1]
        self.ffn = Sequential([
            Dense(self.ff_units, activation='relu'),
            Dropout(self.dropout_rate),
            Dense(embed_dim),
        ])
        super().build(input_shape)

    def call(self, x, training=False):
        attn_output = self.attn(x, x, training=training)
        attn_output = self.attn_norm(x + attn_output)
        ffn_output = self.ffn(attn_output, training=training)
        return self.ffn_norm(attn_output + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "ff_units": self.ff_units,
            "dropout_rate": self.dropout_rate,
        })
        return config


def split_data(x, y, c):

    caseids = np.unique(c)
    ntest = max(1, int(len(caseids) * 0.3))
    caseids_train, caseids_test = caseids[ntest:], caseids[:ntest]

    train_mask, test_mask = np.isin(c, caseids_train), np.isin(c, caseids_test)
    x_train, x_test = x[train_mask], x[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    c_train, c_test= c[train_mask], c[test_mask]

    print('====================================================')
    print(f'Total: {len(caseids)} cases, {len(y)} samples')
    print(f'Train: {len(np.unique(c[train_mask]))} cases, {len(y_train)} samples')
    print(f'Train cases: {len(caseids_train)}, Test cases: {len(caseids_test)}')
    print('====================================================')
    return x_train, x_test, y_train, y_test

def fft_band_power(x_data, srate=128):
    """
    Handles both (N, T) and (N, C, T) shapes.
    Returns: (N, bands) or (N, C, bands)
    """
    if x_data.ndim == 2:
        x_data = x_data[:, np.newaxis, :]  # Add dummy channel dimension

    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    }

    fft_vals = np.fft.rfft(x_data, axis=-1)
    fft_mag = np.abs(fft_vals)
    freqs = np.fft.rfftfreq(x_data.shape[-1], d=1/srate)

    band_features = []
    for (low, high) in bands.values():
        idx = np.where((freqs >= low) & (freqs <= high))[0]
        band_power = fft_mag[:, :, idx].mean(axis=-1)
        band_features.append(band_power)

    return np.stack(band_features, axis=-1)  # (N, C, bands)

def build_model(x_train):
    _, T, F = x_train.shape      # T = #bands, F = #channels
    inputs = Input(shape=(T, F))

    units=128
    dropout=0

    x = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
    x = Conv1D(filters=64, kernel_size=3, activation='relu')(x)

    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(filters=128, kernel_size=3, activation='relu')(x)
    x = Conv1D(filters=128, kernel_size=3, activation='relu')(x)

    x = MaxPooling1D(pool_size=2)(x)
    x = PositionalEmbedding(sequence_length=x.shape[1])(x)
    x = TransformerBlock(num_heads=4, key_dim=units, ff_units=128, dropout_rate=dropout)(x)

    x = Conv1D(filters=256, kernel_size=3, activation='relu')(x)
    x = Conv1D(filters=256, kernel_size=3, activation='relu')(x)

    x = MaxPooling1D(pool_size=2)(x)

    x = Bidirectional(LSTM(units*2, return_sequences=True))(x)
    x = TransformerBlock(num_heads=4, key_dim=units*2, ff_units=512, dropout_rate=dropout)(x)

    x = GlobalAveragePooling1D()(x)

    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs, name="functional_bis_model")

    return model

def evaluate_models(x, y, c, model_dict):

    X_train, X_test, y_train, y_test = split_data(x,y,c)
    results = {}

    for name, model in model_dict.items():
        print(f"Training: {name}...")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        corr = np.corrcoef(y_test, y_pred)[0, 1]  # Pearson correlation

        results[name] = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'Corr': corr
        }

        errors = y_test - y_pred
        abs_errors = np.abs(errors)
        plt.figure(figsize=(6, 6))
        sc = plt.scatter(y_test, y_pred, c=abs_errors, s=2, cmap='viridis', alpha=0.6)
        plt.xlabel('Actual BIS')
        plt.ylabel('Predicted BIS')
        plt.title(f'Colored Error Scatter Plot of {model}')
        plt.colorbar(sc, label='Absolute Error')
        plt.plot([0, max(y_test)], [0, max(y_test)], 'r--')
        plt.grid(True)
        plt.show()


    return results

def windowed_band_power(x_data, srate=128, win_sec=0.25, hop_sec=0.125):
    """
    Accepts (N, T) or (N, C, T). Returns (N, n_windows, C*bands).
    Uses sliding windows to compute per-window band power time series.
    """
    if x_data.ndim == 2:        # (N, T) -> (N, 1, T)
        x_data = x_data[:, np.newaxis, :]
    N, C, T = x_data.shape

    bands = [
        (0.5, 4),   # delta
        (4, 8),     # theta
        (8, 13),    # alpha
        (13, 30),   # beta
        (30, 45),   # gamma
    ]
    B = len(bands)

    win_len = int(round(win_sec * srate))   # e.g., 0.25s -> 32 samples
    hop_len = int(round(hop_sec * srate))   # e.g., 0.125s -> 16 samples
    if win_len <= 0 or hop_len <= 0 or win_len > T:
        raise ValueError("Bad window/hop settings for the given signal length.")

    # number of windows
    n_windows = 1 + (T - win_len) // hop_len

    # precompute frequency bins for window length
    freqs = np.fft.rfftfreq(win_len, d=1.0/srate)

    # allocate output
    out = np.empty((N, n_windows, C * B), dtype=np.float32)

    # compute band power per window
    for i_win in range(n_windows):
        start = i_win * hop_len
        end   = start + win_len
        segment = x_data[:, :, start:end]                  # (N, C, win_len)

        # FFT magnitude per window
        fft_vals = np.fft.rfft(segment, axis=-1)
        fft_mag  = np.abs(fft_vals)                        # (N, C, FreqBins)

        # average mag within each band
        features = []
        for (low, high) in bands:
            idx = (freqs >= low) & (freqs <= high)
            bp = fft_mag[:, :, idx].mean(axis=-1)          # (N, C)
            features.append(bp)

        # stack bands then flatten channels: (N, C, B) -> (N, C*B)
        features = np.stack(features, axis=-1)             # (N, C, B)
        features = features.reshape(N, C * B)              # (N, C*B)
        out[:, i_win, :] = features

    return out  # (N, n_windows, C*B)

x_data= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/Data Files/x_data_without_filter_150.joblib")
y_data = load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/Data Files/b_data_without_filter_150.joblib")
c_data= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/Data Files/c_data_without_filter_150.joblib")

print(f"Available memory: {psutil.virtual_memory().available / (1024 ** 3):.2f} GB")

""""# 1) Prepare both inputs
print(x_data.shape)
x_fourier    = fft_band_power(x_data)                       # (N, C, B)
print(x_fourier.shape)
x_flat       = x_fourier.reshape(x_fourier.shape[0], -1)    # (N, C*B)
print(x_flat.shape)
x_fft_ts     = np.transpose(x_fourier, (0, 2, 1))           # (N, B, C)
print(x_fft_ts.shape)

# 2) Build & compile your DNN on the 3D shape
dnn = build_model(x_fft_ts)                                 # sees (B, C)
dnn.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 3) Define two separate model‐sets
sk_models  = {
    "Linear Regression":      LinearRegression(),
    #"Gaussian Process":       GaussianProcessRegressor(),
    # "SVM (Linear Kernel)":  SVR(kernel='linear'),
}
dl_models  = {"DNN": dnn}

# 4a) Evaluate sklearn on the 2D flat features
results_sk = evaluate_models(x_flat, y_data, c_data, sk_models)

# 4b) Evaluate your DNN on the 3D FFT “time‐series”
results_dl = evaluate_models(x_fft_ts, y_data, c_data, dl_models)

# 5) Combine into one DataFrame
df_results = pd.concat({
    "FFT (sklearn)"  : pd.DataFrame(results_sk).T,
    "FFT (DNN)"      : pd.DataFrame(results_dl).T,
})
print(df_results.round(4))"""

print(x_data.shape)
x_band_ts = windowed_band_power(x_data, srate=128, win_sec=0.25, hop_sec=0.125)
print(x_band_ts.shape)
x_flat = x_band_ts.reshape(x_band_ts.shape[0], -1)
print(x_flat.shape)

# Your DNN expects 3D (timesteps, features). Feed it the time series directly:
dnn = build_model(x_band_ts)     # sees (timesteps=~63, features=5*C)
dnn.compile(optimizer='adam', loss='mse', metrics=['mae'])

sk_models  = {
    "Linear Regression":      LinearRegression(),
}
dl_models  = {"DNN": dnn}

# 4a) Evaluate sklearn on the 2D flat features
results_sk = evaluate_models(x_flat, y_data, c_data, sk_models)

# 5) Combine into one DataFrame
df_results = pd.concat({
    "FFT (sklearn)"  : pd.DataFrame(results_sk).T,
})
print(df_results.round(4))


