import numpy as np
from joblib import load, dump
from keras import Sequential, Model
from keras.src.saving import register_keras_serializable, load_model
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib.pyplot as plt
import os
from keras.src.layers import Input, Dense, Dropout, Conv1D, Bidirectional, LayerNormalization, LSTM, MaxPooling1D, GlobalAveragePooling1D, MultiHeadAttention
from keras.src.optimizers import Adam
from keras.src.callbacks import EarlyStopping
from keras.src.optimizers.schedules import CosineDecayRestarts
import tensorflow as tf
from keras import layers
from VitalDBDataset import VitalDBDataset

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

def build_model(x_train, y_train):

    x_train = x_train.reshape((-1, 1024, 1))
    inputs = Input(shape=(1024, 1))
    units = 128
    dropout = 0.0

    x = Conv1D(64, 3, activation='relu', padding='same')(inputs)
    x = Conv1D(64, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(2)(x)

    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(2)(x)

    # positional embedding needs the current sequence length:
    seq_len = x.shape[1]
    x = PositionalEmbedding(sequence_length=seq_len)(x)
    x = TransformerBlock(num_heads=4, key_dim=units, ff_units=128, dropout_rate=dropout)(x)

    x = Conv1D(256, 3, activation='relu', padding='same')(x)
    x = Conv1D(256, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(2)(x)  # now sequence length is 128

    x = Bidirectional(LSTM(units * 2, return_sequences=True))(x)
    x = TransformerBlock(num_heads=4, key_dim=units * 2, ff_units=512, dropout_rate=dropout)(x)

    x = GlobalAveragePooling1D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64,  activation='relu')(x)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs, name="functional_bis_model_1024")

    # learning‐rate schedule
    lr_schedule = CosineDecayRestarts(
        initial_learning_rate=1e-4,
        first_decay_steps=5,
        t_mul=2.0,
        m_mul=0.9,
        alpha=1e-6
    )

    model.compile(
        optimizer=Adam(learning_rate=lr_schedule),
        loss='huber',
        metrics=['mae']
    )

    model.fit(
        x_train, y_train,
        #validation_data=(x_test, y_test),
        epochs=15,
        batch_size=32,
        callbacks=[EarlyStopping(monitor='val_mae', patience=6, restore_best_weights=True)],
        verbose=1
    )

    # v4 => 10 epochs => 4.14
    # v3 => 15 epochs => 3.88
    # v2 => 20 epochs => 4.18?
    # v1 => 30 epochs => 4.3?
    #model.save('raw_model_v3.keras')

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


dataset=VitalDBDataset(max_cases=150)
x_train_raw = dataset.x_train
x_test_raw = dataset.x_test
y_train_raw = dataset.y_train
y_test_raw = dataset.y_test

print(x_test_raw.shape)

raw_model = build_model(x_train_raw, y_train_raw)
#raw_model= load_model('raw_model_v3.keras')
evaluate_model(raw_model, x_test_raw, y_test_raw)


""""📊 Test MAE: 4.2560
🧮 MSE: 35.0886
📏 RMSE: 5.9236
📈 Correlation coefficient: 0.8222
⌐ R² score: 0.6422"""