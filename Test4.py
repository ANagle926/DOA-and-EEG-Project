import numpy as np
import torch
from joblib import load
from keras import Sequential, Model
from keras.src.saving import load_model, register_keras_serializable
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import os
from keras.src.layers import Input, Dense, Dropout, Conv1D, Bidirectional, LayerNormalization, LSTM, MaxPooling1D, GlobalAveragePooling1D, MultiHeadAttention
from keras.src.optimizers import Adam
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau
import keras
from scipy.signal import periodogram
from sklearn.inspection import permutation_importance
import tensorflow as tf
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore

"""def train_ensemble(x_train, y_train, x_test, y_test, n_models=5):
    preds = []
    for i in range(n_models):
        print(f"\n🔁 Training model {i + 1}/{n_models}")

        # Vary dropout and GaussianNoise upward
        dropout = np.random.choice([0, 0.1, 0.2])
        units = np.random.choice([64, 128])
        batch_size = np.random.choice([32, 64])
        lr = np.random.choice([1e-4])

        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(1024, 3)),
            Conv1D(128, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),

            Bidirectional(LSTM(units * 2, return_sequences=True)),
            LayerNormalization(),
            Dropout(dropout),
            LSTM(units, return_sequences=True),

            TransformerBlock(num_heads=4, key_dim=units, ff_units=256, dropout_rate=dropout),
            TransformerBlock(num_heads=4, key_dim=units, ff_units=256, dropout_rate=dropout),

            GlobalAveragePooling1D(),

            Dense(256, activation='relu'),
            LayerNormalization(),
            Dropout(dropout),
            Dense(64, activation='relu'),
            Dense(1)
        ])

        model.compile(
            optimizer=Adam(lr),
            loss=keras.losses.Huber(delta=1.0),
            metrics=['mae']
        )

        callbacks = [
            EarlyStopping(monitor='val_mae', patience=10, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='val_mae', factor=0.5, patience=5, min_lr=1e-8, verbose=0)
        ]

        model.fit(
            x_train, y_train,
            validation_data=(x_test, y_test),
            epochs=80,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        model.save(f'ensemble_model_{i}.keras')
        y_pred = model.predict(x_test, verbose=1)
        preds.append(y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        print(f"📌 Model {i + 1} MAE: {mae:.4f}")


    ensemble_preds = np.mean(preds, axis=0)
    ensemble_mae = mean_absolute_error(y_test, ensemble_preds)
    print(f"\n📊 Ensemble MAE: {ensemble_mae:.4f}")
    return ensemble_preds, ensemble_mae"""


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from sklearn.base import BaseEstimator, RegressorMixin

@register_keras_serializable()
class TransformerBlock(keras.layers.Layer):
    def __init__(self, num_heads, key_dim, ff_units, dropout_rate, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)  # Pass kwargs to parent constructor
        self.attn = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.ffn = Sequential([
            Dense(ff_units, activation='relu'),
            Dropout(dropout_rate),
            Dense(key_dim)
        ])
        self.attn_norm = LayerNormalization()
        self.ffn_norm = LayerNormalization()
        self.dropout = Dropout(dropout_rate)

    def call(self, x, training=False):
        assert x.ndim == 3, "Expected input shape (batch, seq_len, features)"
        attn_output = self.attn(x, x, training=training)
        attn_output = self.attn_norm(x + attn_output)
        ffn_output = self.ffn(attn_output, training=training)
        ffn_output = self.ffn_norm(attn_output + ffn_output)
        return ffn_output

    def call(self, x, training=False):
        assert x.ndim == 3, "Expected input shape (batch, seq_len, features)"
        attn_output = self.attn(x, x, training=training)
        attn_output = self.attn_norm(x + attn_output)
        ffn_output = self.ffn(attn_output, training=training)
        ffn_output = self.ffn_norm(attn_output + ffn_output)
        return ffn_output

class KerasRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        return self  # Already trained

    def predict(self, X):
        if hasattr(X, 'numpy'):
            X = X.numpy()
        X = np.asarray(X).astype(np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.model.predict(X).reshape(-1)

def get_device(force_cpu=False):
    if force_cpu or not torch.cuda.is_available():
        print("⚙️ Using CPU")
        return torch.device("cpu")
    print("⚡ Using GPU")
    return torch.device("cuda")

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

def create_bis_regressor_model(x_train, y_train, x_test, y_test):
    inputs = Input(shape=(1024, 3))

    x = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
    x = Conv1D(filters=128, kernel_size=3, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Bidirectional(LSTM(256, return_sequences=True))(x)
    x = LayerNormalization()(x)
    x = LSTM(128, return_sequences=True)(x)

    x = TransformerBlock(num_heads=4, key_dim=128, ff_units=256, dropout_rate=0)(x)
    x = TransformerBlock(num_heads=4, key_dim=128, ff_units=256, dropout_rate=0)(x)

    x = GlobalAveragePooling1D()(x)

    x = Dense(256, activation='relu')(x)
    x = LayerNormalization()(x)
    x = Dropout(0)(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs, name="functional_bis_model")

    model.compile(
        optimizer=Adam(0.00001),
        loss=keras.losses.Huber(delta=1.0),
        metrics=['mae']
    )

    callbacks = [
        EarlyStopping(monitor='val_mae', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_mae', factor=0.5, patience=5, min_lr=1e-8, verbose=1)
    ]

    model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=80,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    model.save("eeg_regressor.keras")
    return model

def train_ensemble(x_train, y_train, x_test, y_test, n_models=5):
    preds = []

    for i in range(n_models):
        print(f"\n🔁 Training model {i + 1}/{n_models}")

        dropout = np.random.choice([0, 0.1, 0.2])
        units = np.random.choice([64, 128])
        batch_size = np.random.choice([32, 64])
        lr = np.random.choice([1e-4])

        inputs = Input(shape=(1024, 3))

        x = Conv1D(64, kernel_size=3, activation='relu')(inputs)
        x = Conv1D(128, kernel_size=3, activation='relu')(x)
        x = MaxPooling1D(pool_size=2)(x)

        x = Bidirectional(LSTM(units * 2, return_sequences=True))(x)
        x = LayerNormalization()(x)
        if dropout > 0:
            x = Dropout(dropout)(x)

        x = LSTM(units, return_sequences=True)(x)

        x = TransformerBlock(num_heads=4, key_dim=units, ff_units=256, dropout_rate=dropout)(x)
        x = TransformerBlock(num_heads=4, key_dim=units, ff_units=256, dropout_rate=dropout)(x)

        x = GlobalAveragePooling1D()(x)
        x = Dense(256, activation='relu')(x)
        x = LayerNormalization()(x)
        if dropout > 0:
            x = Dropout(dropout)(x)
        x = Dense(64, activation='relu')(x)
        outputs = Dense(1)(x)

        model = Model(inputs=inputs, outputs=outputs)

        model.compile(
            optimizer=Adam(lr),
            loss=keras.losses.Huber(delta=1.0),
            metrics=['mae']
        )

        callbacks = [
            EarlyStopping(monitor='val_mae', patience=10, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='val_mae', factor=0.5, patience=5, min_lr=1e-8, verbose=0)
        ]

        model.fit(
            x_train, y_train,
            validation_data=(x_test, y_test),
            epochs=80,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        model.save(f'ensemble_model_{i}.keras')
        y_pred = model.predict(x_test, verbose=1)
        preds.append(y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        print(f"📌 Model {i + 1} MAE: {mae:.4f}")

    ensemble_preds = np.mean(preds, axis=0)
    ensemble_mae = mean_absolute_error(y_test, ensemble_preds)
    print(f"\n📊 Ensemble MAE: {ensemble_mae:.4f}")
    return ensemble_preds, ensemble_mae

def perform_permutation_importance(x_test, y_test, model):

    x_test_flat = x_test[:2000].reshape(x_test.shape[0], -1)

    x_test_clean = np.asarray(x_test_flat).astype(np.float32)
    #x_test_clean = tf.convert_to_tensor(x_test) if isinstance(x_test, np.ndarray) else x_test
    #x_test_clean = x_test_clean.numpy() if hasattr(x_test_clean, 'numpy') else x_test_clean
    x_test_clean = np.asarray(x_test_clean).astype(np.float32)

    # 2. Wrap the model
    wrapped_model = KerasRegressorWrapper(model)
    print("created model")

    # 3. Run permutation importance
    result = permutation_importance(
        wrapped_model,
        x_test_clean,  # ✅ use cleaned input here
        y_test,
        n_repeats=10,
        random_state=42,
        scoring='neg_mean_absolute_error'
    )
    print("ran permutation importance")

    # 4. Generate feature names for display
    num_features = x_test_clean.shape[1]
    feature_names = [f'f{i}' for i in range(num_features)]
    print("generated feature names")

    # 5. Display importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': result.importances_mean,
        'std': result.importances_std
    }).sort_values(by='importance', ascending=False)

    print(importance_df)
    plot_feature_importance(importance_df)

    return importance_df

def plot_feature_importance(importance_df, top_n=20):
    top_features = importance_df.head(top_n)[::-1]  # reverse for horizontal bar chart

    plt.figure(figsize=(10, 6))
    plt.barh(top_features['feature'], top_features['importance'], xerr=top_features['std'])
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title(f'Top {top_n} Important Features')
    plt.tight_layout()
    plt.grid(True)
    plt.show()

def prune_features_and_remap_dataset(x_train, x_test, importance_df, threshold=0.0):
    importance_df["index"] = importance_df["feature"].str.extract(r"f(\d+)").astype(int)
    important_features = importance_df[importance_df["importance"] > threshold]["index"].values
    important_features.sort()
    x_train_pruned = x_train[:, important_features]
    x_test_pruned = x_test[:, important_features]

    return x_train_pruned, x_test_pruned, important_features

def perform_integrated_gradients_feature_importance(model, x_data):
    def score_function(output):
        return output[:, 0]  # Directly return regression outputs


    modifier = ReplaceToLinear()
    saliency = Saliency(model, model_modifier=modifier, clone=True)

    # === Batched Saliency Computation ===
    batch_size = 16  # adjust if OOM persists
    saliency_maps = []

    for i in range(0, x_data.shape[0], batch_size):
        batch = x_data[i:i + batch_size]
        saliency_batch = saliency(score_function, batch)
        saliency_maps.append(saliency_batch)

    saliency_map = np.concatenate(saliency_maps, axis=0)
    # === End Batch Processing ===

    saliency_map = tf.convert_to_tensor(saliency_map)
    if tf.executing_eagerly():
        saliency_map = saliency_map.numpy()
    else:
        saliency_map = tf.compat.v1.Session().run(saliency_map)

    saliency_flat = saliency_map.reshape(saliency_map.shape[0], -1)
    feature_importance = np.mean(np.abs(saliency_flat), axis=0)

    importance_df = pd.DataFrame({
        'feature': [f'f{i}' for i in range(len(feature_importance))],
        'importance': feature_importance
    }).sort_values(by='importance', ascending=False)
    plot_feature_importance(importance_df)

    return importance_df


dataset=load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/Data Files/dataset_twenty_cases_SEGLENMID.joblib")

x_train, y_train = dataset.x_train, dataset.y_train
x_test, y_test = dataset.x_test, dataset.y_test
c_test= dataset.c_test
c_train= dataset.c_train

#analyze_dataset(x_train, y_train, x_test, y_test)

print("x_train_raw shape:", x_train.shape)
print("x_test_raw shape:", x_test.shape)

#model = create_bis_regressor_model(x_train, y_train, x_test, y_test)
model=load_model("eeg_regressor.keras")
evaluate_model(model, x_test, y_test)

importance_df = perform_integrated_gradients_feature_importance(model, x_test[:2000])
importance_df.to_csv("feature_importance.csv", index=False)

x_train, x_test, kept_feature_indices = prune_features_and_remap_dataset(
    x_train,
    x_test,
    importance_df,
    threshold=0.01
)

np.save("kept_feature_indices.npy", kept_feature_indices)

model = create_bis_regressor_model(x_train, y_train, x_test, y_test)
evaluate_model(model, x_test, y_test)
ensemble_preds, ensemble_mae = train_ensemble(x_train, y_train, x_test, y_test, n_models=5)

