from joblib import load
from keras import Sequential, Model
from keras.src.saving import load_model, register_keras_serializable
import matplotlib.pyplot as plt
from keras.src.layers import Input, Dense, Dropout, Conv1D, Bidirectional, LayerNormalization, LSTM, MaxPooling1D, \
    GlobalAveragePooling1D, MultiHeadAttention, Concatenate, Add, Activation, BatchNormalization
import tensorflow as tf
from keras import layers
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, r2_score
import seaborn as sns
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score, classification_report

# 0.8 accuracy

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

def apply_feature_pruning(x_data, important_channels, timestep_masks):
    """Prunes both channels and timesteps within channels"""
    # 1. Select important channels
    x_pruned = x_data[:, :, important_channels]

    # 2. Apply temporal masks to each selected channel
    for i, orig_ch in enumerate(important_channels):
        mask = timestep_masks[orig_ch]
        # Zero out unimportant timesteps in this channel
        x_pruned[:, ~mask, i] = 0.0  # Replace with baseline if needed

    return x_pruned

def fast_predict_with_ensemble(x_test, y_test=None, meta_model_path="Saved Model Versions/gbrt_model.pkl"):
    # Load meta-model and top-k indices
    meta_data = joblib.load(meta_model_path)
    gbrt_model = meta_data["gbrt_model"]
    topk_idx = meta_data["topk_idx"]

    # Load only top-k models and predict
    topk_preds = []
    for i in topk_idx:
        model = load_model(f'Saved Model Versions/Ensemble/ensemble_model_v4__{i}.keras')
        y_pred = model.predict(x_test, verbose=0)
        topk_preds.append(y_pred)
    P_test = np.hstack(topk_preds)

    # Sanity check
    assert P_test.shape[1] == gbrt_model.n_features_in_

    # Meta-model prediction
    meta_preds = gbrt_model.predict(P_test)

    # Optionally compute MAE
    mae = None
    if y_test is not None:
        mae = mean_absolute_error(y_test.flatten(), meta_preds.flatten())
        print(f"Meta-Ensemble GBRT MAE: {mae:.4f}")

    return meta_preds, mae

def threshold_classifier(y_pred_reg, thresholds=(40, 60)):
    """
    Convert BIS regression predictions into 3-class discrete labels.
    - BIS < 40     → Class 0 (AD)
    - 40–60        → Class 1 (AO)
    - >60          → Class 2 (AL)
    """
    t1, t2 = thresholds
    return np.where(y_pred_reg < t1, 0,
                    np.where(y_pred_reg <= t2, 1, 2))

def evaluate_threshold_based_classifier(y_true_cont, y_pred_cont, thresholds=(40, 60)):
    y_true_cls = threshold_classifier(y_true_cont, thresholds)
    y_pred_cls = threshold_classifier(y_pred_cont, thresholds)

    acc = accuracy_score(y_true_cls, y_pred_cls)
    print(f"Threshold-based Classification Accuracy: {acc:.4f}\n")
    print("Classification Report:\n", classification_report(y_true_cls, y_pred_cls, digits=4))

    cm = confusion_matrix(y_true_cls, y_pred_cls)
    return acc, cm



dataset = load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/Data Files/dataset_twenty_cases_SEGLENMID.joblib")
x_test, y_test = dataset.x_test, dataset.y_test
x_train, y_train = dataset.x_train, dataset.y_train

important_channels, timestep_masks = joblib.load("Saved Model Versions/Pruning/pruning_artifacts_v2.joblib")
x_train_pruned = apply_feature_pruning(x_train, important_channels, timestep_masks)
x_test_pruned = apply_feature_pruning(x_test, important_channels, timestep_masks)

meta_preds, mae = fast_predict_with_ensemble(x_test_pruned, y_test)

# Evaluate using domain thresholds (fixed)
acc, cm = evaluate_threshold_based_classifier(y_test, meta_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
