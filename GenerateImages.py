import numpy as np
from keras import Sequential, Model
from keras.src.saving import register_keras_serializable
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib.pyplot as plt
import os
from keras.src.layers import Input, Dense, Dropout, Conv1D, Bidirectional, LayerNormalization, LSTM, MaxPooling1D, GlobalAveragePooling1D, MultiHeadAttention
from keras.src.optimizers import Adam
from keras.src.callbacks import EarlyStopping
import keras
from keras.src.optimizers.schedules import CosineDecayRestarts
import tensorflow as tf
from keras import layers
from VB2 import VB2+

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

def evaluate_model(model, x_test, y_test):

    pred_test = model.predict(x_test).flatten()
    test_mae = mean_absolute_error(y_test, pred_test)
    mse = mean_squared_error(y_test, pred_test)
    rmse = np.sqrt(mse)
    corr = np.corrcoef(y_test, pred_test)[0, 1]
    r2 = r2_score(y_test, pred_test)

    print(f"\n\U0001f4ca Test MAE: {test_mae:.4f}")
    print(f"🧮 MSE: {mse:.4f}")
    print(f"📏 RMSE: {rmse:.4f}")
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
    for i in range(8):
        plt.subplot(3, 1, i+1)
        plt.plot(saliency_maps[0,:,i])
        plt.title(f"Temporal Importance - Channel {i}")
        plt.xlabel("Timesteps")
        plt.ylabel("Importance")

    plt.tight_layout()
    plt.show()

def perform_integrated_gradients_feature_importance(model, x_data):
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

dataset=VB2(max_cases=10)
x_test, y_test = dataset.x_test, dataset.y_test
x_train, y_train = dataset.x_train, dataset.y_train
print(x_train.shape)
model = build_model(x_train, y_train, x_test, y_test)
model.save("model_all_imfs_test.keras")

perform_integrated_gradients_feature_importance(model, x_train[:2000])