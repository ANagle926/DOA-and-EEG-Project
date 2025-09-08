import numpy as np
from joblib import load, dump
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

    # (num_samples, 1024) → (num_samples, 1024, 1)
    x_train = x_train.reshape((-1, 1024, 1))
    x_test  = x_test.reshape((-1, 1024, 1))
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
        validation_data=(x_test, y_test),
        epochs=80,
        batch_size=32,
        callbacks=[EarlyStopping(monitor='val_mae', patience=6, restore_best_weights=True)],
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


def _ensure_3d(x):
    """Return (N, T, C). Accepts (N, T) or (N, T, 1)."""
    x = np.asarray(x)
    if x.ndim == 2:
        x = x[..., None]
    elif x.ndim != 3:
        raise ValueError(f"x must be (N,T) or (N,T,1); got shape {x.shape}")
    if x.shape[-1] != 1:
        raise ValueError(f"Single-channel expected; got C={x.shape[-1]}")
    return x

def optimized_integrated_gradients_1d(model, baseline, input_data, m_steps=50, sample_batch_size=16):

    #Memory-safe Integrated Gradients for 1D single-channel inputs

    x = _ensure_3d(input_data).astype(np.float32)   # (N, T, 1)
    baseline = _ensure_3d(baseline).astype(np.float32)  # (1, T, 1)
    N, T, _ = x.shape
    alphas = tf.linspace(0.0, 1.0, m_steps + 1)  # (m+1,)

    all_attrs = []

    for i in range(0, N, sample_batch_size):
        batch = tf.convert_to_tensor(x[i:i+sample_batch_size])  # (B, T, 1)

        # Repeat baseline and inputs along the IG path dimension
        baseline_rep = tf.repeat(tf.convert_to_tensor(baseline), repeats=m_steps + 1, axis=0)
        batch_attrs = []

        for j in range(batch.shape[0]):
            x1 = batch[j:j+1]  # (1, T, 1)
            x_rep = tf.repeat(x1, repeats=m_steps + 1, axis=0)

            # Interpolate along the path
            interpolated = baseline_rep + tf.reshape(alphas, (-1, 1, 1)) * (x_rep - baseline_rep)

            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(interpolated)
                preds = model(interpolated)
                outputs = tf.reshape(preds[..., 0], (-1,))

            grads = tape.gradient(outputs, interpolated)
            grads = tf.cast(grads, tf.float32)

            # Trapezoidal average of gradients along the path
            avg_grads = tf.reduce_mean((grads[:-1] + grads[1:]) / 2.0, axis=0) # (T, 1)

            # Attribution: (x - baseline) * avg_grads
            attr = (x1 - baseline) * avg_grads                                 # (1, T, 1)
            batch_attrs.append(attr[0])                                        # (T, 1)

        batch_attrs = tf.stack(batch_attrs, axis=0)     # (B, T, 1)
        all_attrs.append(batch_attrs)

    all_attrs = tf.concat(all_attrs, axis=0)            # (N, T, 1)
    return tf.squeeze(all_attrs, axis=-1).numpy()       # (N, T)

def visualize_saliency_1d(saliency_maps, sample_idx=0, title="Temporal Importance (IG)"):
    """
    saliency_maps: (N, T) array of attributions.
    """
    s = saliency_maps[sample_idx]
    plt.figure(figsize=(12, 4))
    plt.plot(s)
    plt.title(title + f" — sample {sample_idx}")
    plt.xlabel("Timesteps")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.show()

def IG_temporal_pruning(model, x_data, timestep_percentile=50, aggregate="mean", m_steps=50, sample_batch_size=16, return_all=False):

    #Compute a single global timestep mask via Integrated Gradients.

    X = _ensure_3d(x_data)
    N, T, _ = X.shape
    baseline = np.zeros((1, T, 1), dtype=np.float32)

    saliency_maps = optimized_integrated_gradients_1d(
        model=model,
        baseline=baseline,
        input_data=X,
        m_steps=m_steps,
        sample_batch_size=sample_batch_size
    )

    visualize_saliency_1d(saliency_maps=saliency_maps)

    # Aggregate absolute importance across samples to get a single timeline
    if aggregate == "mean":
        timestep_importance = np.mean(np.abs(saliency_maps), axis=0)  # (T,)
    elif aggregate == "median":
        timestep_importance = np.median(np.abs(saliency_maps), axis=0)
    else:
        raise ValueError("aggregate must be 'mean' or 'median'")

    threshold = np.percentile(timestep_importance, timestep_percentile)
    mask = timestep_importance >= threshold                           # (T,)

    if return_all:
        return mask, saliency_maps, timestep_importance
    return mask

def apply_temporal_pruning(x_data, mask, fill_value=0.0):
    """
    Zero (or fill) out unimportant timesteps using a boolean mask over T.
    Accepts (N,T) or (N,T,1).
    Returns pruned array with the same shape as input.
    """
    X = np.asarray(x_data)
    if X.ndim == 2:
        X_out = X.copy()
        X_out[:, ~mask] = fill_value
    elif X.ndim == 3:
        X_out = X.copy()
        X_out[:, ~mask, :] = fill_value
    else:
        raise ValueError(f"x_data must be (N,T) or (N,T,1); got {X.shape}")
    return X_out


def create_GBRT(preds, y_true):

    y_true = y_true.flatten()
    P = np.hstack(preds)

    model = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)

    model.fit(P, y_true)
    final_predictions = model.predict(P)

    dump(model, "Files/Saved Model Files/GBRT/gbrt_model_test.pkl")

    return final_predictions, model

def create_ensemble(x_train, y_train, x_test, y_test, n_models=5):

    inputs = Input(shape=x_train.shape[1:])
    preds = []

    for i in range(n_models):
        print(f"\n🔁 Training model {i + 1}/{n_models}")

        units = int(np.random.choice([64, 128]))
        dropout = float(np.random.choice([0, 0.1]))
        batch_size = int(np.random.choice([16, 32, 64]))
        lr = float(np.random.choice([0.0001, 0.0005]))

        print(f"🧪 units={units}, dropout=({dropout}, batch_size={batch_size}, lr={lr}")

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

        model.save(f'ensemble_model_150_test.{i}.keras')
        y_pred = model.predict(x_test, verbose=1)
        preds.append(y_pred)

    return preds



def split_data(x, y, c):

    caseids = np.unique(c)
    ntest = max(1, int(len(caseids) * 0.3))
    caseids_train, caseids_test = caseids[ntest:], caseids[:ntest]

    train_mask, test_mask = np.isin(c, caseids_train), np.isin(c, caseids_test)
    x_train, x_test = x[train_mask], x[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    print('====================================================')
    print(f'Total: {len(caseids)} cases, {len(y)} samples')
    print(f'Train: {len(np.unique(c[train_mask]))} cases, {len(y_train)} samples')
    print(f'Train cases: {len(caseids_train)}, Test cases: {len(caseids_test)}')
    print('====================================================')
    return x_train, x_test, y_train, y_test


#loading raw EEG data
x= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/Data Files/x_data_without_filter_150.joblib")
y= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/Data Files/b_data_without_filter_150.joblib")
c= load( "/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/Data Files/c_data_without_filter_150.joblib")
x_train_raw, x_test_raw, y_train_raw, y_test_raw= split_data(x, y, c)

#model trained on raw EEG data
raw_model = build_model(x_train_raw, y_train_raw, x_test_raw, y_test_raw)
evaluate_model(raw_model, x_test_raw, y_test_raw)

#pruning raw data
mask, saliency_maps, importance = IG_temporal_pruning(
    raw_model,
    x_train_raw[:2000],
    timestep_percentile=60,
    aggregate="mean",
    return_all=True
)
visualize_saliency_1d(saliency_maps, sample_idx=0)
x_train_raw_pruned = apply_temporal_pruning(x_train_raw, mask, fill_value=0.0)
x_test_raw_pruned= apply_temporal_pruning(x_test_raw, mask, fill_value=0.0)

#model trained on raw pruned EEG data
raw_pruned_model = build_model(x_train_raw_pruned, y_train_raw, x_test_raw_pruned, y_test_raw)
evaluate_model(raw_pruned_model, x_test_raw_pruned, y_test_raw)

n_models=5


#ensemble with raw data
raw_preds= create_ensemble( x_train_raw, y_train_raw, x_test_raw, y_test_raw, n_models=n_models)
raw_meta_test_preds, raw_gbrt_model = create_GBRT(raw_preds, y_test_raw)

mae = mean_absolute_error(y_test_raw, raw_meta_test_preds)
mse = mean_squared_error(y_test_raw, raw_meta_test_preds)
corr = np.corrcoef(y_test_raw, raw_meta_test_preds)[0, 1]
r2 = r2_score(y_test_raw, raw_meta_test_preds)

print(f"🔍 Meta-Ensemble GBRT MAE on Test Set: {mae:.4f}")
print(f"🔍 Meta-Ensemble GBRT MSE on Test Set: {mse:.4f}")
print(f"🔍 Meta-Ensemble GBRT CORR on Test Set: {corr:.4f}")
print(f"🔍 Meta-Ensemble GBRT RSQUARED on Test Set: {r2:.4f}")


#ensemble with raw pruned data
raw_pruned_preds= create_ensemble( x_train_raw_pruned, y_train_raw, x_test_raw_pruned, y_test_raw, n_models=n_models)
raw_pruned_meta_test_preds, raw_pruned_gbrt_model = create_GBRT(raw_pruned_preds, y_test_raw)

mae = mean_absolute_error(y_test_raw, raw_pruned_meta_test_preds)
mse = mean_squared_error(y_test_raw, raw_pruned_meta_test_preds)
corr = np.corrcoef(y_test_raw, raw_pruned_meta_test_preds)[0, 1]
r2 = r2_score(y_test_raw, raw_pruned_meta_test_preds)

print(f"🔍 Meta-Ensemble GBRT MAE on Test Set: {mae:.4f}")
print(f"🔍 Meta-Ensemble GBRT MSE on Test Set: {mse:.4f}")
print(f"🔍 Meta-Ensemble GBRT CORR on Test Set: {corr:.4f}")
print(f"🔍 Meta-Ensemble GBRT RSQUARED on Test Set: {r2:.4f}")