import numpy as np
from joblib import load
from keras import Sequential
from keras.src.saving import register_keras_serializable, load_model
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error
import os
from keras.src.layers import Dense, Dropout, LayerNormalization, MultiHeadAttention
import tensorflow as tf
from keras import layers

#plotting EEG, true BIS, and predicted BIS on same graph

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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

def load_ensemble(x_test, y_test, c_test, n_models):
    preds = []

    for i in range(n_models):

        model= load_model(f'/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/Files/Saved Model Files/Ensemble/ensemble_model_150_raw_without_pruning.{i}.keras')
        y_pred = model.predict(x_test, verbose=1)
        plot_data(c_test, x_test, y_test, y_pred)
        preds.append(y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        print("mae:", mae)

    return preds

def split_data(x, y, c):

    caseids = np.unique(c)
    ntest = max(1, int(len(caseids) * 0.3))
    caseids_train, caseids_test = caseids[ntest:], caseids[:ntest]

    train_mask, test_mask = np.isin(c, caseids_train), np.isin(c, caseids_test)
    x_train, x_test = x[train_mask], x[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    c_train, c_test = c[train_mask], c[test_mask]

    print('====================================================')
    print(f'Total: {len(caseids)} cases, {len(y)} samples')
    print(f'Train: {len(np.unique(c[train_mask]))} cases, {len(y_train)} samples')
    print(f'Train cases: {len(caseids_train)}, Test cases: {len(caseids_test)}')
    print('====================================================')
    return x_train, x_test, y_train, y_test, c_train, c_test

def plot_data(c_test, x_test, y_test, preds, fs=128, win_sec=8, segment_sec=2000, seed=0):
    rng = np.random.default_rng(seed)

    # -- pick one random case --
    caseid = rng.choice(np.unique(c_test))
    mask = (c_test == caseid)

    x_case = x_test[mask]                    # (num_windows, 1024)
    y_case = np.asarray(y_test[mask]).reshape(-1)   # (num_windows,)
    p_case = np.asarray(preds[mask]).reshape(-1)    # (num_windows,)

    # -- build continuous EEG & per-sample BIS series (step per 8s window) --
    eeg_wave = x_case.reshape(-1)            # 1D EEG for this case
    samples_per_win = int(fs * win_sec)      # 128 * 8 = 1024

    bis_series  = np.repeat(y_case, samples_per_win)[:eeg_wave.size]
    pred_series = np.repeat(p_case, samples_per_win)[:eeg_wave.size]

    # -- choose a random zoomed segment --
    N = eeg_wave.size
    seg_N = min(int(segment_sec * fs), N)
    start = 0 if N == seg_N else rng.integers(0, N - seg_N)
    end = start + seg_N

    t_rel   = np.arange(seg_N) / fs
    eeg_seg = eeg_wave[start:end]
    bis_seg = bis_series[start:end]
    pred_seg = pred_series[start:end]

    mae_seg = float(np.mean(np.abs(bis_seg - pred_seg)))

    # -- plot --
    fig, ax1 = plt.subplots(figsize=(20, 4))
    ax1.plot(t_rel, eeg_seg, linewidth=0.9, label='EEG')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('EEG amplitude')
    ax1.grid(True, alpha=0.2)

    ax2 = ax1.twinx()
    ax2.plot(t_rel, bis_seg,  label='BIS (true)', drawstyle='steps-post', linewidth=5, color = "crimson")
    ax2.plot(t_rel, pred_seg, label='BIS (pred)', drawstyle='steps-post', linewidth=5, color ="gold")
    ax2.set_ylim(0, 100)
    ax2.set_ylabel('BIS (0–100)')

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

    plt.title(f'Case {caseid} — {segment_sec}s segment | MAE={mae_seg:.2f}')
    plt.tight_layout()
    plt.show()

x= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/Files/Data Files/x_data_without_filter_150.joblib")
y= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/Files/Data Files/b_data_without_filter_150.joblib")
c= load( "/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/Files/Data Files/c_data_without_filter_150.joblib")
x_train, x_test, y_train, y_test, c_train, c_test= split_data(x, y, c)

n_models=5
preds =load_ensemble(x_test, y_test, c_test, n_models)