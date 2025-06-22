import keras
from joblib import load
from keras import Sequential, Model
from keras.src.callbacks import EarlyStopping
from keras.src.losses import CategoricalCrossentropy
from keras.src.metrics import Precision, Recall, AUC
from keras.src.optimizers import Adam
from keras.src.optimizers.schedules import CosineDecayRestarts
from keras.src.saving import load_model, register_keras_serializable
import matplotlib.pyplot as plt
import os
from keras.src.layers import Input, Dense, Dropout, Conv1D, Bidirectional, LayerNormalization, LSTM, MaxPooling1D, \
    GlobalAveragePooling1D, MultiHeadAttention, Concatenate, Add, Activation, BatchNormalization
import tensorflow as tf
from keras import layers
import joblib
from keras.src.utils import to_categorical
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, r2_score
import seaborn as sns
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score, classification_report
from scipy.optimize import minimize

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

#0.76 accuracy


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

def convert_to_hard_labels(y_array):
    labels = []
    for value in y_array:
        if value < 40:
            labels.append(0)
        elif value < 60:
            labels.append(1)
        else:
            labels.append(2)
    return np.array(labels)


def weighted_topk_ensemble(preds, val_maes, k=3):
    val_maes = np.array(val_maes)
    preds = np.array(preds)
    topk_idx = np.argsort(val_maes)[:k]
    topk_maes = val_maes[topk_idx]
    topk_preds = preds[topk_idx]
    weights = 1 / (topk_maes + 1e-8)
    weights /= weights.sum()
    ensemble_preds = np.average(topk_preds, axis=0, weights=weights)
    return ensemble_preds, topk_idx, weights

def create_ensemble(x_train, y_train, x_test, y_test, y_train_class, y_test_class, n_models=5):
    reg_preds_list, cls_preds_list, val_maes = [], [], []
    inputs = Input(shape=x_train.shape[1:])

    for i in range(n_models):
        print(f"\n🔁 Training model {i + 1}/{n_models}")

        # Model definition
        x = Conv1D(64, 3, activation='relu')(inputs)
        x = Conv1D(64, 3, activation='relu')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(128, 3, activation='relu')(x)
        x = Conv1D(128, 3, activation='relu')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = PositionalEmbedding(sequence_length=500)(x)
        x = TransformerBlock(num_heads=4, key_dim=64, ff_units=256, dropout_rate=0.0)(x)
        x = Conv1D(256, 3, activation='relu')(x)
        x = Conv1D(256, 3, activation='relu')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Bidirectional(LSTM(128, return_sequences=True))(x)
        x = TransformerBlock(num_heads=4, key_dim=128, ff_units=512, dropout_rate=0.0)(x)
        x = GlobalAveragePooling1D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        reg_output = Dense(1, name="regression_output")(x)
        cls_output = Dense(3, activation='softmax', name="class_output")(x)

        model = Model(inputs=inputs, outputs=[reg_output, cls_output])
        lr_schedule = CosineDecayRestarts(0.0001, first_decay_steps=5, t_mul=2.0, m_mul=0.9, alpha=1e-6)

        model.compile(
            optimizer=Adam(learning_rate=lr_schedule),
            loss={"regression_output": "huber", "class_output": CategoricalCrossentropy(label_smoothing=0.1)},
            loss_weights={"regression_output": 1.0, "class_output": 0.5},
            metrics={
                "regression_output": ["mae"],
                "class_output": ["accuracy", Precision(), Recall()]
            }
        )

        model.fit(
            x_train,
            {"regression_output": y_train, "class_output": y_train_class},
            validation_data=(x_test, {"regression_output": y_test, "class_output": y_test_class}),
            epochs=80,
            batch_size=32,
            callbacks=[EarlyStopping(monitor='val_mae', patience=6, restore_best_weights=True)],
            verbose=1
        )

        model.save(f'ensemble_multihead_model_v1__{i}.keras')
        reg_pred, cls_pred = model.predict(x_test, verbose=0)
        reg_preds_list.append(reg_pred.flatten())
        cls_preds_list.append(cls_pred)
        val_maes.append(mean_absolute_error(y_test.flatten(), reg_pred.flatten()))

    # Get ensemble weights and top-k
    reg_preds_array = np.array(reg_preds_list)
    cls_preds_array = np.array(cls_preds_list)
    val_maes = np.array(val_maes)
    topk_idx = np.argsort(val_maes)[:3]
    weights = 1 / (val_maes[topk_idx] + 1e-8)
    weights /= weights.sum()

    # Regression ensemble
    reg_ensemble = np.average(reg_preds_array[topk_idx], axis=0, weights=weights)

    # Classification ensemble (soft-voting)
    cls_ensemble = np.average(cls_preds_array[topk_idx], axis=0, weights=weights)

    # Classify from both
    reg_based_classes = regression_to_class(reg_ensemble)
    cls_based_classes = np.argmax(cls_ensemble, axis=1)
    true_classes = np.argmax(y_test_class, axis=1)

    # Print final metrics
    print("\n📊 Final Regression Metrics:")
    print("MAE:", mean_absolute_error(y_test.flatten(), reg_ensemble))
    print("R2:", r2_score(y_test.flatten(), reg_ensemble))
    print("Corr:", np.corrcoef(y_test.flatten(), reg_ensemble)[0, 1])

    print("\n📊 Final Classification Metrics:")
    print("Reg→Class Accuracy:", accuracy_score(true_classes, reg_based_classes))
    print("Softmax Ensemble Accuracy:", accuracy_score(true_classes, cls_based_classes))
    print("Classification Report:\n", classification_report(true_classes, cls_based_classes))

    return reg_ensemble, cls_ensemble, topk_idx

def regression_to_class(preds):
    """Convert regression predictions into 3-class labels."""
    bins = np.digitize(preds, bins=[40, 60])  # class 0: <40, class 1: 40–60, class 2: >60
    return bins


def discretize_and_onehot(y_array):
    class_labels = regression_to_class(y_array)
    return to_categorical(class_labels, num_classes=3)


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import joblib

def meta_ensemble(preds, y_true, val_maes, top_k=3):
    top_indices = np.argsort(val_maes)[:top_k]
    print(f"🔢 Using top-{top_k} models: indices {top_indices}")

    # Make sure each preds[i] is (12000,) or (12000, 1)
    pred_list = [preds[i].reshape(-1, 1) for i in top_indices]  # shape: (12000, 1)
    reg_features = np.hstack(pred_list)  # Final shape: (12000, top_k)

    # Check shape consistency
    assert reg_features.shape[0] == y_true.shape[0], "Mismatch in sample counts"

    gbrt = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
    gbrt.fit(reg_features, y_true.flatten())
    final_preds = gbrt.predict(reg_features)

    # Save for inference later
    joblib.dump({"gbrt_model": gbrt, "topk_idx": top_indices}, "gbrt_model.pkl")

    print("✅ GBRT MAE:", mean_absolute_error(y_true.flatten(), final_preds))
    return final_preds, gbrt




dataset = load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/Data Files/dataset_twenty_cases_SEGLENMID.joblib")
x_test, y_test = dataset.x_test, dataset.y_test
x_train, y_train = dataset.x_train, dataset.y_train

important_channels, timestep_masks = joblib.load("Saved Model Versions/Regressor/Pruning/pruning_artifacts_v2.joblib")
x_train_pruned = apply_feature_pruning(x_train, important_channels, timestep_masks)
x_test_pruned = apply_feature_pruning(x_test, important_channels, timestep_masks)

y_train_class = convert_to_hard_labels(y_train)
y_test_class = convert_to_hard_labels(y_test)
y_train_class = to_categorical(y_train_class, num_classes=3)
y_test_class = to_categorical(y_test_class, num_classes=3)


n_models=6
reg_preds_list, val_maes, topk_idx  = create_ensemble(x_train_pruned, y_train, x_test_pruned, y_test, y_train_class, y_test_class, n_models=6)

meta_preds, gbrt_model = meta_ensemble(
    reg_preds_list, y_test, val_maes, top_k=3
)