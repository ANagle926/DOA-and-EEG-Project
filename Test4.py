import numpy as np
import torch
from joblib import load
from keras import Sequential, Model
from keras.src.saving import load_model, register_keras_serializable
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
from keras.src.layers import Input, Dense, Dropout, Conv1D, Bidirectional, LayerNormalization, LSTM, MaxPooling1D, \
    GlobalAveragePooling1D, MultiHeadAttention, Concatenate, Add, Activation
from keras.src.optimizers import Adam
from keras.src.callbacks import EarlyStopping
import keras
from scipy.signal import periodogram
from keras.src.optimizers.schedules import CosineDecayRestarts
import tensorflow as tf
from keras import layers
import joblib
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import GradientBoostingRegressor



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
class AttentionPooling1D(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.attention_weights = self.add_weight(
            name="attention_weights",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True,
        )

    def call(self, inputs):
        scores = tf.matmul(inputs, self.attention_weights)  # (batch, time, 1)
        scores = tf.nn.softmax(scores, axis=1)
        return tf.reduce_sum(inputs * scores, axis=1)

    def get_config(self):
        return super().get_config()
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


def get_device(force_cpu=False):
    if force_cpu or not torch.cuda.is_available():
        print("⚙️ Using CPU")
        return torch.device("cpu")
    print("⚡ Using GPU")
    return torch.device("cuda")


def build_model(x_train, y_train, x_test, y_test):
    inputs = Input(shape=x_train.shape[1:])
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

    initial_learning_rate = 0.0001
    first_decay_steps = 5
    t_mul = 2.0
    m_mul = 0.9

    lr_schedule = CosineDecayRestarts(
        initial_learning_rate=initial_learning_rate,
        first_decay_steps=first_decay_steps,
        t_mul=t_mul,
        m_mul=m_mul,
        alpha=1e-6  # minimum learning rate
    )

    model.compile(
        optimizer=Adam(learning_rate=lr_schedule),
        loss=keras.losses.Huber(delta=1.0),
        metrics=['mae']
    )

    callbacks = [
        EarlyStopping(monitor='val_mae', patience=6, restore_best_weights=True, verbose=1),
    ]

    model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=80,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    return model


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

def evaluate_model(model, x_test, y_test):
    #if x_test.ndim == 2:
    #    x_test = np.expand_dims(x_test, axis=-1)

    pred_test = model.predict(x_test).flatten()
    test_mae = mean_absolute_error(y_test, pred_test)
    corr = np.corrcoef(y_test, pred_test)[0, 1]
    r2 = r2_score(y_test, pred_test)

    print(f"\n\U0001f4ca Test MAE: {test_mae:.4f}")
    print(f"\U0001f4c8 Correlation coefficient: {corr:.4f}")
    print(f"\u2310 R² score: {r2:.4f}")

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


def optimized_integrated_gradients(model, baseline, input_data, m_steps=50, sample_batch_size=8):
    """
    Safer version of Integrated Gradients: loops over samples in small batches,
    and interpolates per-sample to reduce memory footprint.
    """
    import tensorflow as tf

    baseline = tf.convert_to_tensor(baseline, dtype=tf.float32)
    input_data = tf.convert_to_tensor(input_data, dtype=tf.float32)

    n_samples = input_data.shape[0]
    alphas = tf.linspace(0.0, 1.0, m_steps + 1)

    all_attributions = []

    for i in range(0, n_samples, sample_batch_size):
        batch = input_data[i:i+sample_batch_size]
        batch_attributions = []

        for j in range(batch.shape[0]):
            x = batch[j:j+1]
            baseline_repeated = tf.repeat(baseline, repeats=m_steps + 1, axis=0)
            x_repeated = tf.repeat(x, repeats=m_steps + 1, axis=0)

            interpolated = baseline_repeated + tf.reshape(alphas, (-1, 1, 1)) * (x_repeated - baseline_repeated)

            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(interpolated)
                predictions = model(interpolated)
                outputs = predictions[:, 0]

            grads = tape.gradient(outputs, interpolated)
            grads = tf.reshape(grads, [m_steps + 1] + list(x.shape[1:]))
            avg_grads = tf.reduce_mean((grads[:-1] + grads[1:]) / 2.0, axis=0)

            attr = (x - baseline) * avg_grads
            batch_attributions.append(attr)

        all_attributions.append(tf.concat(batch_attributions, axis=0))

    return tf.concat(all_attributions, axis=0).numpy()

def visualize_saliency(saliency_maps):
    # Temporal importance visualization
    plt.figure(figsize=(12, 6))

    # First sample, first 3 channels
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.plot(saliency_maps[0,:,i])
        plt.title(f"Temporal Importance - Channel {i}")
        plt.xlabel("Timesteps")
        plt.ylabel("Importance")

    plt.tight_layout()
    plt.show()

def perform_integrated_gradients_feature_importance(model, x_data, channel_threshold=0.1, timestep_percentile=50):
    # Baseline: Zero-valued EEG signals
    baseline = np.zeros((1, 1024, 3))

    saliency_maps = optimized_integrated_gradients(
        model=model,
        baseline=baseline,
        input_data=x_data,
        m_steps=50,
        sample_batch_size=16
    )

    visualize_saliency(saliency_maps=saliency_maps)

    # 1. Channel (IMF) selection
    channel_importance = np.sum(np.abs(saliency_maps), axis=(0,1))
    total_importance = np.sum(channel_importance)
    important_channels = np.where(channel_importance/total_importance >= channel_threshold)[0]

    if len(important_channels) == 0:
        important_channels = np.array([np.argmax(channel_importance)])

    # 2. Temporal pruning within channels
    timestep_masks = {}
    for ch in important_channels:
        # Average importance across samples for this channel
        timestep_importance = np.mean(np.abs(saliency_maps[:,:,ch]), axis=0)

        # Dynamic threshold based on percentile
        threshold = np.percentile(timestep_importance, timestep_percentile)
        timestep_masks[ch] = timestep_importance >= threshold

    return important_channels, timestep_masks

def apply_feature_pruning(x_data, important_channels, timestep_masks):
    """Prunes both channels and timesteps within channels"""
    # 1. Select important channels
    x_pruned = x_data[:, :, important_channels]

    # 2. Apply temporal masks to each selected channel
    for i, orig_ch in enumerate(important_channels):
        mask = timestep_masks[orig_ch]
        # Zero out unimportant timesteps in this channel
        x_pruned[:, ~mask, i] = 0.0  # Replace with baseline if needed

    return x_pruned\


def weighted_topk_ensemble(preds, val_maes, k=3):
    val_maes = np.array(val_maes)
    preds = np.array(preds)  # shape: (n_models, n_samples)

    # Step 1: Get indices of top-k models (lowest MAEs)
    topk_idx = np.argsort(val_maes)[:k]

    # Step 2: Extract top-k MAEs and predictions
    topk_maes = val_maes[topk_idx]
    topk_preds = preds[topk_idx]  # shape: (k, n_samples)

    # Step 3: Inverse MAE weighting
    weights = 1 / (topk_maes + 1e-8)
    weights /= weights.sum()  # Normalize

    # Step 4: Weighted average of top-k predictions
    ensemble_preds = np.average(topk_preds, axis=0, weights=weights)

    return ensemble_preds, topk_idx, weights

def meta_ensemble(preds, y_true, method="ridge", val_maes=None, top_k=None):
    """
    Train a second-level model (meta-learner) on base model predictions.

    Parameters:
        preds: list of np.arrays of shape (n_samples, 1)
        y_true: true targets, shape (n_samples,)
        method: "ridge" or "gbrt"
        val_maes: validation MAEs for top_k selection
        top_k: number of top models to use (optional)

    Returns:
        final_predictions: predictions from meta-model
        model: trained meta-model
    """

    y_true = y_true.flatten()

    if top_k is not None:
        if val_maes is None:
            raise ValueError("val_maes must be provided when using top_k")
        top_indices = np.argsort(val_maes)[:top_k]
        preds = [preds[i] for i in top_indices]
        print(f"🔢 Using top-{top_k} models: indices {top_indices}")

    P = np.hstack(preds)

    if method == "ridge":
        model = RidgeCV(alphas=np.logspace(-3, 3, 10), cv=5)
    elif method == "gbrt":
        model = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
    else:
        raise ValueError("Unknown method")

    model.fit(P, y_true)
    final_predictions = model.predict(P)
    return final_predictions, model

def create_ensemble(x_train, y_train, x_test, y_test, n_models=5):

    preds = []
    val_maes =[]

    inputs = Input(shape=x_train.shape[1:])

    for i in range(n_models):
        print(f"\n🔁 Training model {i + 1}/{n_models}")

        units = int(np.random.choice([64, 128]))
        dropout = float(np.random.choice([0]))
        batch_size = int(np.random.choice([32, 100]))
        lr = float(np.random.choice([0.0001]))

        print(f"🧪 units={units}, dropout=({dropout}, batch_size={batch_size}, lr={lr}")

        x = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
        x = Conv1D(filters=64, kernel_size=3, activation='relu')(x)

        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(filters=128, kernel_size=3, activation='relu')(x)
        x = Conv1D(filters=128, kernel_size=3, activation='relu')(x)

        x = MaxPooling1D(pool_size=2)(x)
        x = PositionalEmbedding(sequence_length=500)(x)
        x = TransformerBlock(num_heads=4, key_dim=units, ff_units=256, dropout_rate=dropout)(x)

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

        initial_learning_rate = lr
        first_decay_steps = 5
        t_mul = 2.0
        m_mul = 0.9

        lr_schedule = CosineDecayRestarts(
            initial_learning_rate=initial_learning_rate,
            first_decay_steps=first_decay_steps,
            t_mul=t_mul,
            m_mul=m_mul,
            alpha=1e-6  # minimum learning rate
        )

        model.compile(
            optimizer=Adam(learning_rate=lr_schedule),
            loss=keras.losses.Huber(delta=1.0),
            metrics=['mae']
        )

        callbacks = [
            EarlyStopping(monitor='val_mae', patience=6, restore_best_weights=True, verbose=1),
        ]

        model.fit(
            x_train, y_train,
            validation_data=(x_test, y_test),
            epochs=80,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        model.save(f'ensemble_model_v4__{i}.keras')
        y_pred = model.predict(x_test, verbose=1)
        preds.append(y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        val_maes.append(mae)

        print(f"📌 Model {i + 1} MAE: {mae:.4f}")

    ensemble_preds, topk_idx, weights = weighted_topk_ensemble(preds, val_maes, k=3)

    # Flatten for metric calculation
    ensemble_preds = ensemble_preds.flatten()
    y_test = y_test.flatten()

    # Compute performance metrics
    ensemble_mae = mean_absolute_error(y_test, ensemble_preds)
    ensemble_r2 = r2_score(y_test, ensemble_preds)
    ensemble_corr = np.corrcoef(y_test, ensemble_preds)[0, 1]

    # Display results
    print(f"\n📊 Top-{len(topk_idx)} Weighted Ensemble MAE: {ensemble_mae:.4f}")
    print(f"📈 Correlation coefficient: {ensemble_corr:.4f}")
    print(f"📐 R² score: {ensemble_r2:.4f}")
    print(f"🏆 Used model indices: {topk_idx}")
    print(f"📊 Normalized weights: {weights}")

    return ensemble_preds, ensemble_mae, ensemble_r2, ensemble_corr

def create_GBRT(n_models, x_test, y_test):
    preds = []
    val_maes = []

    for i in range(n_models):
        model = load_model(f'ensemble_model_v4__{i}.keras')
        y_pred = model.predict(x_test, verbose=1)
        preds.append(y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        val_maes.append(mae)
        print(f"📌 Model {i + 1} MAE: {mae:.4f}")

    # Get top-k indices
    top_k = 3
    topk_idx = np.argsort(val_maes)[:top_k]
    topk_preds = [preds[i] for i in topk_idx]

    # Train meta-model
    meta_preds_gbrt, gbrt_model = meta_ensemble(topk_preds, y_test, method="gbrt")

    P_test = np.hstack(topk_preds)  # Make sure topk_preds is a list of (n_samples, 1) arrays

    # Save everything
    joblib.dump({
        "gbrt_model": gbrt_model,
        "topk_idx": topk_idx,
        "topk_preds_test": P_test
    }, "gbrt_model.pkl")


    # Use only top-k preds for test input
    P_test = np.hstack([preds[i] for i in topk_idx])
    assert P_test.shape[1] == gbrt_model.n_features_in_, \
        f"Feature mismatch: expected {gbrt_model.n_features_in_}, got {P_test.shape[1]}"

    meta_test_preds = gbrt_model.predict(P_test)
    mae = mean_absolute_error(y_test, meta_test_preds)
    print(f"🔍 Meta-Ensemble GBRT MAE on Test Set: {mae:.4f}")


dataset=load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/Data Files/dataset_twenty_cases_SEGLENMID.joblib")

x_train, y_train = dataset.x_train, dataset.y_train
x_test, y_test = dataset.x_test, dataset.y_test
c_test= dataset.c_test
c_train= dataset.c_train

#analyze_dataset(x_train, y_train, x_test, y_test)

print("x_train_raw shape:", x_train.shape)
print("x_test_raw shape:", x_test.shape)


#model = build_model(x_train, y_train, x_test, y_test)
#model.save("eeg_regressor_v4.keras")
model = load_model("eeg_regressor_v4.keras")
evaluate_model(model, x_test, y_test)


"""# Compute important features using training data
important_channels, timestep_masks = perform_integrated_gradients_feature_importance(
    model, x_train[:2000],
    channel_threshold=0.15,
    timestep_percentile=60
)

joblib.dump((important_channels, timestep_masks), "pruning_artifacts_v4.joblib")
print("✅ Saved pruning artifacts to pruning_artifacts_v4.joblib")

# Prune both datasets using same features (prevents data leakage)
x_train_pruned = apply_feature_pruning(x_train, important_channels, timestep_masks)
x_test_pruned = apply_feature_pruning(x_test, important_channels, timestep_masks)

assert x_train_pruned.shape[2] == len(important_channels), \
    "Channel count mismatch after pruning"
assert x_test_pruned.shape[1:] == x_train_pruned.shape[1:], \
    "Train/test shape mismatch"""

important_channels, timestep_masks = joblib.load("pruning_artifacts_v2.joblib")
x_train_pruned = apply_feature_pruning(x_train, important_channels, timestep_masks)
x_test_pruned = apply_feature_pruning(x_test, important_channels, timestep_masks)
print(x_train_pruned.shape)


#model = build_model(x_train_pruned, y_train, x_test_pruned, y_test)
#model.save("eeg_regressor_pruned_v4.keras")
model = load_model("eeg_regressor_pruned_v4.keras")
evaluate_model(model, x_test_pruned, y_test)

n_models=6
ensemble_preds, ensemble_mae, ensemble_r2, ensemble_corr= create_ensemble( x_train_pruned, y_train, x_test_pruned, y_test, n_models=n_models)
create_GBRT(n_models=n_models, x_test=x_test_pruned, y_test=y_test)