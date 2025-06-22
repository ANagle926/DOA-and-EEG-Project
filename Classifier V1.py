import numpy as np
import torch
from imblearn.over_sampling import SMOTE
from joblib import load
from keras import Sequential, Model
from keras.src.losses import CategoricalCrossentropy
from keras.src.metrics import Precision, Recall, AUC
from keras.src.saving import load_model, register_keras_serializable
from sklearn.metrics import mean_absolute_error, r2_score, log_loss, accuracy_score, f1_score
import matplotlib.pyplot as plt
import os
from keras.src.layers import Input, Dense, Dropout, Conv1D, Bidirectional, LayerNormalization, LSTM, MaxPooling1D, \
    GlobalAveragePooling1D, MultiHeadAttention, Concatenate, Add, Activation, BatchNormalization
from keras.src.optimizers import Adam
from keras.src.callbacks import EarlyStopping
from scipy.signal import periodogram
from keras.src.optimizers.schedules import CosineDecayRestarts
import tensorflow as tf
from keras import layers
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import GradientBoostingClassifier


#0.83 accuracy

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

def build_model(x_train, y_train, x_test, y_test, class_weight_dict):
    inputs = Input(shape=x_train.shape[1:])
    units = 32
    dropout = 0.4

    x = Conv1D(filters=32, kernel_size=3, activation='relu')(inputs)
    x = Conv1D(filters=32, kernel_size=3, activation='relu')(x)

    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(filters=64, kernel_size=3, activation='relu')(x)
    x = Conv1D(filters=64, kernel_size=3, activation='relu')(x)

    x = MaxPooling1D(pool_size=2)(x)
    x = PositionalEmbedding(sequence_length=x.shape[1])(x)
    x = TransformerBlock(num_heads=4, key_dim=units, ff_units=128, dropout_rate=dropout)(x)
    x = LayerNormalization()(x)


    #x = Conv1D(filters=128, kernel_size=3, activation='relu')(x)
    #x = Conv1D(filters=128, kernel_size=3, activation='relu')(x)

    #x = MaxPooling1D(pool_size=2)(x)

    x = Bidirectional(LSTM(units*2, return_sequences=True))(x)
    x = TransformerBlock(num_heads=4, key_dim=units*2, ff_units=512, dropout_rate=dropout)(x)
    x = LayerNormalization()(x)

    x = GlobalAveragePooling1D()(x)

    #x = Dense(256, activation='relu')(x)
    #x=Dropout(dropout)(x)
    #x = Dense(128, activation='relu')(x)
    #x=Dropout(dropout)(x)

    x = Dense(64, activation='relu')(x)
    outputs = Dense(3, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs, name="functional_bis_model")

    lr_schedule = CosineDecayRestarts(
        initial_learning_rate=0.001,
        first_decay_steps=5,
        t_mul=2.0,
        m_mul=0.9,
        alpha=1e-6
    )

    model.compile(
        optimizer=Adam(learning_rate=lr_schedule),
        loss=CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy', Precision(), Recall(), AUC()]
    )


    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),
    ]

    model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        class_weight=class_weight_dict,
        epochs=80,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    return model

def preprocess_data(x_train, y_train, y_test):
    """Convert labels to categories, apply SMOTE, and one-hot encode."""
    """y_train_cat = convert_to_categories(y_train)
    y_test_cat = convert_to_categories(y_test)

    # Save original shape
    orig_shape = x_train.shape

    # Flatten x_train for SMOTE
    x_train_flat = x_train.reshape(x_train.shape[0], -1)

    # Apply SMOTE for class balancing
    #smote = SMOTE(random_state=42)
    smote = SMOTE(sampling_strategy={0:10000, 1:15000, 2:8000})  # Imbalanced but realistic

    x_train_resampled, y_train_resampled = smote.fit_resample(x_train_flat, y_train_cat)

    # Reshape back to (samples, timesteps, channels)
    x_train_resampled = x_train_resampled.reshape(-1, orig_shape[1], orig_shape[2])

    # Convert labels to one-hot encoding
    y_train_cat = to_categorical(y_train_resampled, num_classes=3)
    y_test_cat = to_categorical(y_test_cat, num_classes=3)

    print("x_train_resampled shape:", x_train_resampled.shape)
    print("y_train_cat shape:", y_train_cat.shape)

    return x_train_resampled, y_train_cat, y_test_cat"""
    y_train_soft = convert_to_soft_labels(y_train)
    y_test_soft = convert_to_soft_labels(y_test)

    class_weight_dict = {
        0: 3.0,
        1: 1.0,
        2: 6.0
    }

    return x_train, y_train_soft, y_test_soft, class_weight_dict

def convert_to_soft_labels(y_array):
    soft_labels = []
    for value in y_array:
        if value < 35:
            soft_labels.append([1.0, 0.0, 0.0])  # class 0
        elif value < 42:
            soft_labels.append([0.7, 0.3, 0.0])  # reduce class 0 pull
        elif value < 50:
            soft_labels.append([0.2, 0.7, 0.1])  # sharper center bias
        elif value < 60:
            soft_labels.append([0.1, 0.6, 0.3])  # more class 2 guidance
        elif value < 68:
            soft_labels.append([0.0, 0.2, 0.8])  # very class 2 heavy
        else:
            soft_labels.append([0.0, 0.05, 0.95])  # class 2 confident
    return np.array(soft_labels)

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
    y_pred = np.argmax(model.predict(x_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    acc = accuracy_score(y_true, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    print(classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
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
                target_class = tf.argmax(predictions, axis=1)
                outputs = tf.gather_nd(predictions, tf.stack([tf.range(predictions.shape[0]), target_class], axis=1))


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

    return x_pruned

def weighted_topk_ensemble_classification(preds, val_losses, k=3):
    val_losses = np.array(val_losses)
    preds = np.array(preds)  # shape: (n_models, n_samples, n_classes)

    # Step 1: Get indices of top-k models (lowest loss)
    topk_idx = np.argsort(val_losses)[:k]

    # Step 2: Extract top-k losses and predictions
    topk_losses = val_losses[topk_idx]
    topk_preds = preds[topk_idx]  # shape: (k, n_samples, n_classes)

    # Step 3: Inverse loss weighting
    weights = 1 / (topk_losses + 1e-8)
    weights /= weights.sum()

    # Step 4: Weighted average of class probabilities
    ensemble_preds = np.tensordot(weights, topk_preds, axes=([0], [0]))  # shape: (n_samples, n_classes)

    return ensemble_preds, topk_idx, weights


def meta_ensemble_classification(preds, y_true, method="logreg", val_losses=None, top_k=None):
    y_true_cls = np.argmax(y_true, axis=1)  # Convert one-hot to labels

    if top_k is not None:
        if val_losses is None:
            raise ValueError("val_losses must be provided when using top_k")
        top_indices = np.argsort(val_losses)[:top_k]
        preds = [preds[i] for i in top_indices]
        print(f"🔢 Using top-{top_k} models: indices {top_indices}")

    P = np.hstack(preds)  # shape: (n_samples, k * n_classes)

    if method == "logreg":
        model = LogisticRegressionCV(cv=5, multi_class='multinomial', max_iter=1000)
    elif method == "gbrt":
        model = GradientBoostingClassifier(n_estimators=100, max_depth=3)
    else:
        raise ValueError("Unknown method")

    model.fit(P, y_true_cls)
    final_predictions = model.predict_proba(P)
    return final_predictions, model

def create_ensemble(x_train, y_train, x_test, y_test, class_weight_dict, n_models=5):
    preds = []
    val_losses = []
    accuracies=[]

    for i in range(n_models):
        print(f"\n🔁 Training model {i + 1}/{n_models}")

        units = int(np.random.choice([32, 64]))
        dropout = float(np.random.choice([0.3, 0.6]))
        batch_size = int(np.random.choice([16, 32, 128]))
        lr = float(np.random.choice([0.0005, 0.001, 0.003]))

        print(f"🧪 units={units}, dropout={dropout}, batch_size={batch_size}, lr={lr}")

        inputs = Input(shape=x_train.shape[1:])
        x = Conv1D(filters=32, kernel_size=3, activation='relu')(inputs)
        x = Conv1D(filters=32, kernel_size=3, activation='relu')(x)

        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(filters=64, kernel_size=3, activation='relu')(x)
        x = Conv1D(filters=64, kernel_size=3, activation='relu')(x)

        x = MaxPooling1D(pool_size=2)(x)
        x = PositionalEmbedding(sequence_length=x.shape[1])(x)
        x = TransformerBlock(num_heads=4, key_dim=units, ff_units=128, dropout_rate=dropout)(x)
        x = LayerNormalization()(x)

        x = Bidirectional(LSTM(units*2, return_sequences=True))(x)
        x = TransformerBlock(num_heads=4, key_dim=units*2, ff_units=512, dropout_rate=dropout)(x)
        x = LayerNormalization()(x)

        x = GlobalAveragePooling1D()(x)

        x = Dense(64, activation='relu')(x)
        outputs = Dense(3, activation='softmax')(x)


        model = Model(inputs=inputs, outputs=outputs, name=f"ensemble_model_{i}")

        lr_schedule = CosineDecayRestarts(
            initial_learning_rate=lr,
            first_decay_steps=5,
            t_mul=2.0,
            m_mul=0.9,
            alpha=1e-6
        )

        model.compile(
            optimizer=Adam(learning_rate=lr_schedule),
            loss=CategoricalCrossentropy(label_smoothing=0.1),
            metrics=['accuracy', Precision(), Recall(), AUC()]
        )

        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True, verbose=1),
        ]

        model.fit(
            x_train, y_train,
            validation_data=(x_test, y_test),
            class_weight=class_weight_dict,
            epochs=80,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        model.save(f'ensemble_model_classifier_v1_{i}.keras')
        y_pred = model.predict(x_test, verbose=1)
        preds.append(y_pred)

        loss = log_loss(np.argmax(y_test, axis=1), y_pred)
        val_losses.append(loss)
        acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
        accuracies.append(acc)

        print(f"📌 Model {i + 1} Log Loss: {loss:.4f}")
        print(f"📌 Model {i + 1} Accuracy: {acc:.4f}")

    preds = np.array(preds)
    ensemble_preds, topk_idx, weights = weighted_topk_ensemble_classification(preds, val_losses, k=3)

    y_true_cls = np.argmax(y_test, axis=1)
    ensemble_preds_cls = np.argmax(ensemble_preds, axis=1)
    ensemble_acc = accuracy_score(y_true_cls, ensemble_preds_cls)
    macro_f1 = f1_score(y_true_cls, ensemble_preds_cls, average='macro')

    ensemble_avg = np.mean(preds, axis=0)
    ensemble_preds_avg_cls = np.argmax(ensemble_avg, axis=1)
    ensemble_acc_avg = accuracy_score(y_true_cls, ensemble_preds_avg_cls)
    macro_f1_avg = f1_score(y_true_cls, ensemble_preds_avg_cls, average='macro')

    print("🧾 Classification Report for top k:\n")
    print(classification_report(y_true_cls, ensemble_preds_cls, digits=3))
    print(f"\n📊 Top-{len(topk_idx)} Weighted Ensemble Accuracy: {ensemble_acc:.4f}")
    print(f"🏆 Ensemble Macro F1 Score: {macro_f1:.4f}")
    print(f"🏆 Used model indices: {topk_idx}")
    print(f"📊 Normalized weights: {weights}")

    print("🧾 Classification Report for average ensemble:\n")
    print(classification_report(y_true_cls, ensemble_preds_avg_cls, digits=3))
    print(f"\n📊 Average Weighted Ensemble Accuracy: {ensemble_acc_avg:.4f}")
    print(f"🏆 Average Ensemble Macro F1 Score: {macro_f1_avg:.4f}")


    return ensemble_preds, ensemble_acc

def create_GBRT(n_models, x_test, y_test):
    preds = []
    val_losses = []

    for i in range(n_models):
        model = load_model(f'ensemble_model_classifier_v1_{i}.keras')
        y_pred = model.predict(x_test, verbose=1)
        preds.append(y_pred)
        loss = log_loss(y_test, y_pred)
        val_losses.append(loss)
        print(f"📌 Model {i + 1} Log Loss: {loss:.4f}")

    top_k = 3
    topk_idx = np.argsort(val_losses)[:top_k]
    topk_preds = [preds[i] for i in topk_idx]

    meta_preds_gbrt, gbrt_model = meta_ensemble_classification(topk_preds, y_test, method="gbrt")
    P_test = np.hstack(topk_preds)

    joblib.dump({
        "gbrt_classifier_model": gbrt_model,
        "topk_idx_classifier": topk_idx,
        "topk_preds_test_classifier": P_test
    }, "Saved Model Versions/Classifier/gbrt_model_classifier.pkl")

    meta_test_preds_cls = np.argmax(meta_preds_gbrt, axis=1)
    y_true_cls = np.argmax(y_test, axis=1)
    acc = accuracy_score(y_true_cls, meta_test_preds_cls)
    print(f"🔍 Meta-Ensemble GBRT Accuracy on Test Set: {acc:.4f}")


dataset=load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/Data Files/dataset_twenty_cases_SEGLENMID.joblib")

x_train, y_train = dataset.x_train, dataset.y_train
x_test, y_test = dataset.x_test, dataset.y_test


#analyze_dataset(x_train, y_train, x_test, y_test)

print("x_train_raw shape:", x_train.shape)
print("x_test_raw shape:", x_test.shape)

x_train, y_train, y_test, class_weight_dict= preprocess_data(x_train, y_train, y_test)

model = build_model(x_train, y_train, x_test, y_test, class_weight_dict)
model.save("eeg_classifier_v2.keras")
#model = load_model("eeg_classifier_v2.keras")

evaluate_model(model, x_test, y_test)


# Compute important features using training data
important_channels, timestep_masks = perform_integrated_gradients_feature_importance(
    model, x_train[:2000],
    channel_threshold=0.15,
    timestep_percentile=60
)

joblib.dump((important_channels, timestep_masks), "pruning_artifacts_classification_v1.joblib")
print("✅ Saved pruning artifacts to pruning_artifacts_classification_v1.joblib")

# Prune both datasets using same features (prevents data leakage)
x_train_pruned = apply_feature_pruning(x_train, important_channels, timestep_masks)
x_test_pruned = apply_feature_pruning(x_test, important_channels, timestep_masks)


"""important_channels, timestep_masks = joblib.load("Saved Model Versions/Pruning/pruning_artifacts_v2.joblib")
x_train_pruned = apply_feature_pruning(x_train, important_channels, timestep_masks)
x_test_pruned = apply_feature_pruning(x_test, important_channels, timestep_masks)
print(x_train_pruned.shape)"""

model = build_model(x_train_pruned, y_train, x_test_pruned, y_test, class_weight_dict)
model.save("eeg_classifier_pruned_v2.keras")
#model = load_model("eeg_classifier_pruned_v1.keras")

evaluate_model(model, x_test_pruned, y_test)

n_models=8
ensemble_preds, ensemble_acc= create_ensemble( x_train_pruned, y_train, x_test_pruned, y_test, class_weight_dict, n_models)
create_GBRT(n_models=n_models, x_test=x_test_pruned, y_test=y_test)