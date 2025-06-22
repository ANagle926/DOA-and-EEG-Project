import joblib
import numpy as np
from joblib import load
from keras import Sequential, Model
from keras.src.callbacks import EarlyStopping
from keras.src.losses import CategoricalCrossentropy, Huber
from keras.src.metrics import Precision, Recall, AUC
from keras.src.optimizers import Adam
from keras.src.optimizers.schedules import CosineDecayRestarts
from keras.src.saving import load_model, register_keras_serializable
from keras.src.layers import Input, Dense, Conv1D, Bidirectional, LSTM, MaxPooling1D, GlobalAveragePooling1D, \
    MultiHeadAttention, LayerNormalization, Dropout
from keras import layers
import tensorflow as tf
from keras.src.utils import to_categorical
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report, ConfusionMatrixDisplay


#accuracy: 0.78

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

def create_regression_ensemble(x_train, y_train, x_test, y_test, n_models=5):
    preds = []
    val_maes = []

    for i in range(n_models):
        print(f"\n🔁 Training model {i + 1}/{n_models}")
        inputs = Input(shape=x_train.shape[1:])

        units = int(np.random.choice([64, 128]))
        dropout = 0.0
        batch_size = int(np.random.choice([32, 100]))
        lr = 0.0001

        x = Conv1D(64, 3, activation='relu')(inputs)
        x = Conv1D(64, 3, activation='relu')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(128, 3, activation='relu')(x)
        x = Conv1D(128, 3, activation='relu')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = PositionalEmbedding(sequence_length=500)(x)
        x = TransformerBlock(num_heads=4, key_dim=units, ff_units=256, dropout_rate=dropout)(x)
        x = Conv1D(256, 3, activation='relu')(x)
        x = Conv1D(256, 3, activation='relu')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Bidirectional(LSTM(units * 2, return_sequences=True))(x)
        x = TransformerBlock(num_heads=4, key_dim=units * 2, ff_units=512, dropout_rate=dropout)(x)
        x = GlobalAveragePooling1D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        output = Dense(1)(x)

        model = Model(inputs, output)

        lr_schedule = CosineDecayRestarts(
            initial_learning_rate=lr,
            first_decay_steps=5,
            t_mul=2.0,
            m_mul=0.9,
            alpha=1e-6
        )

        model.compile(
            optimizer=Adam(learning_rate=lr_schedule),
            loss=Huber(delta=1.0),
            metrics=['mae']
        )

        model.fit(
            x_train, y_train,
            validation_data=(x_test, y_test),
            epochs=80,
            batch_size=batch_size,
            callbacks=[EarlyStopping(monitor='val_mae', patience=6, restore_best_weights=True)],
            verbose=1
        )

        model.save(f'ensemble_model_classification_v4__{i}.keras')
        y_pred = model.predict(x_test, verbose=0)
        preds.append(y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        val_maes.append(mae)

        print(f"\n📌 Model {i + 1} MAE: {mae:.4f}")

    ensemble_preds, topk_idx, weights = weighted_topk_ensemble(preds, val_maes, k=3)

    ensemble_mae = mean_absolute_error(y_test.flatten(), ensemble_preds.flatten())
    ensemble_r2 = r2_score(y_test.flatten(), ensemble_preds.flatten())
    ensemble_corr = np.corrcoef(y_test.flatten(), ensemble_preds.flatten())[0, 1]

    print(f"\n📊 Final Regression Metrics:")
    print(f"MAE: {ensemble_mae:.4f}")
    print(f"R2: {ensemble_r2:.4f}")
    print(f"Corr: {ensemble_corr:.4f}")

    return preds, val_maes, topk_idx

def add_classification_heads(pretrained_model, num_classes=3):
    """
    Freezes all layers in the pretrained regression model and attaches a classification head.
    """

    # Freeze all layers
    for layer in pretrained_model.layers:
        layer.trainable = False

    # Identify the last shared layer (before regression head)
    shared_output = pretrained_model.layers[-2].output  # second to last layer

    # Add new classifier head
    x = Dense(64, activation='relu')(shared_output)
    x = Dropout(0.3)(x)
    class_output = Dense(num_classes, activation='softmax', name="class_output")(x)

    # New model with classification output
    clf_model = Model(inputs=pretrained_model.input, outputs=class_output)

    clf_model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy', Precision(), Recall(), AUC()]
    )

    return clf_model


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



dataset = load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/Data Files/dataset_twenty_cases_SEGLENMID.joblib")
x_test, y_test = dataset.x_test, dataset.y_test
x_train, y_train = dataset.x_train, dataset.y_train

important_channels, timestep_masks = load("Saved Model Versions/Regressor/Pruning/pruning_artifacts_v2.joblib")
x_train_pruned = apply_feature_pruning(x_train, important_channels, timestep_masks)
x_test_pruned = apply_feature_pruning(x_test, important_channels, timestep_masks)

y_train_class = convert_to_hard_labels(y_train)
y_test_class = convert_to_hard_labels(y_test)
y_train_class = to_categorical(y_train_class, num_classes=3)
y_test_class = to_categorical(y_test_class, num_classes=3)

# Step 1: Train regression-only ensemble
reg_preds, val_maes, topk_idx = create_regression_ensemble(
    x_train_pruned, y_train, x_test_pruned, y_test, n_models=6
)


# Step 2: Add classification heads to top-k models
soft_preds = []
for i in topk_idx:
    reg_model = load_model(f'ensemble_model_classification_v4__{i}.keras')
    clf_model = add_classification_heads(reg_model, num_classes=3)
    clf_model.fit(
        x_train_pruned,
        y_train_class,
        validation_data=(x_test_pruned, y_test_class),
        epochs=20,
        batch_size=32,
        verbose=1
    )
    clf_model.save(f"Saved Model Versions/classifier_head_model_{i}.keras")
    soft_preds.append(clf_model.predict(x_test_pruned))

# Step 3: Evaluate ensemble classification
soft_preds = np.array(soft_preds)  # shape: (k, n_samples, 3)
ensemble_softmax_preds = np.mean(soft_preds, axis=0)
ensemble_reg_preds = weighted_topk_ensemble(reg_preds, val_maes, k=3)[0]
ensemble_reg_preds = ensemble_reg_preds.flatten()
ensemble_reg_classes = convert_to_hard_labels(ensemble_reg_preds)

ensemble_softmax_classes = np.argmax(ensemble_softmax_preds, axis=1)
y_true_classes = np.argmax(y_test_class, axis=1)

print("🔍 Softmax Ensemble Accuracy:", accuracy_score(y_true_classes, ensemble_softmax_classes))
print("📋 Classification Report:\n", classification_report(y_true_classes, ensemble_softmax_classes))
ConfusionMatrixDisplay.from_predictions(y_true_classes, ensemble_softmax_classes)
