import numpy as np
from joblib import load, dump
from keras import Sequential, Model
from keras.src.saving import register_keras_serializable
from matplotlib import pyplot
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import os
from keras.src.layers import Input, Dense, Dropout, Conv1D, Bidirectional, LayerNormalization, LSTM, MaxPooling1D, \
    GlobalAveragePooling1D, MultiHeadAttention
from keras.src.optimizers import Adam
from keras.src.callbacks import EarlyStopping
import keras
from keras.src.optimizers.schedules import CosineDecayRestarts
import tensorflow as tf
from keras import layers
from sklearn.ensemble import GradientBoostingRegressor

#Boostrapping the raw EEG model

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


def create_GBRT(preds, y_true):

    y_true = y_true.flatten()
    P = np.hstack(preds)

    model = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)

    model.fit(P, y_true)
    final_predictions = model.predict(P)

    dump(model, "Files/Saved Model Files/GBRT/gbrt_model_bootstrap_raw.pkl")

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

        model.save(f'ensemble_model_150_raw.{i}.keras')
        y_pred = model.predict(x_test, verbose=1)
        preds.append(y_pred)

    return preds


x= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/Data Files/x_data_without_filter_150.joblib")
y= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/Data Files/b_data_without_filter_150.joblib")
c= load( "/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/Data Files/c_data_without_filter_150.joblib")
x_train_raw, x_test_raw, y_train_raw, y_test_raw= split_data(x, y, c)

# configure bootstrap
n_iterations = 15
n_models = 3

# run bootstrap
mae_list = list()
mse_list= list()
rmse_list= list()
corr_list= list()
r2_list= list()

for i in range(n_iterations):

    # prepare train and test sets
    n = x_train_raw.shape[0]
    indices = np.random.choice(n, n, replace=True)
    x_bootstrap = x_train_raw[indices]
    y_bootstrap = y_train_raw[indices]

    # fit model
    preds= create_ensemble(x_train_raw, y_train_raw, x_test_raw, y_test_raw, n_models=n_models)
    meta_test_preds, gbrt_model = create_GBRT(preds, y_test_raw)

    mae = mean_absolute_error(y_test_raw, meta_test_preds)
    mse = mean_squared_error(y_test_raw, meta_test_preds)
    rmse= np.sqrt(mse)
    corr = np.corrcoef(y_test_raw, meta_test_preds)[0, 1]
    r2 = r2_score(y_test_raw, meta_test_preds)

    mae_list.append(mae)
    mse_list.append(mse)
    rmse_list.append(rmse)
    corr_list.append(corr)
    r2_list.append(r2)

    print(f"[{i+1}/{n_iterations}] MAE: {mae:.4f}")
    print(f"[{i+1}/{n_iterations}] MSE: {mse:.4f}")
    print(f"[{i+1}/{n_iterations}] RMSE: {rmse:.4f}")
    print(f"[{i+1}/{n_iterations}] CORR: {corr:.4f}")
    print(f"[{i+1}/{n_iterations}] R2: {r2:.4f}")

# Plot results
pyplot.hist(mae_list, bins=30)
pyplot.xlabel("MAE")
pyplot.ylabel("Frequency")
pyplot.title("Bootstrap Distribution of GBRT Meta-MAE")
pyplot.show()

pyplot.hist(mse_list, bins=30)
pyplot.xlabel("MSE")
pyplot.ylabel("Frequency")
pyplot.title("Bootstrap Distribution of GBRT Meta-MSE")
pyplot.show()

pyplot.hist(rmse_list, bins=30)
pyplot.xlabel("RMSE")
pyplot.ylabel("Frequency")
pyplot.title("Bootstrap Distribution of GBRT Meta-RMSE")
pyplot.show()

pyplot.hist(corr_list, bins=30)
pyplot.xlabel("Corr")
pyplot.ylabel("Frequency")
pyplot.title("Bootstrap Distribution of GBRT Meta-corr")
pyplot.show()

pyplot.hist(r2_list, bins=30)
pyplot.xlabel("r2")
pyplot.ylabel("Frequency")
pyplot.title("Bootstrap Distribution of GBRT Meta-r2")
pyplot.show()

# confidence intervals
alpha = 0.95
lower = np.percentile(mae_list, ((1.0 - alpha) / 2.0) * 100)
upper = np.percentile(mae_list, (alpha + (1.0 - alpha) / 2.0) * 100)
print(f"{alpha*100:.1f}% confidence interval for MAE: {lower:.4f} to {upper:.4f}")

# confidence intervals
lower = np.percentile(mse_list, ((1.0 - alpha) / 2.0) * 100)
upper = np.percentile(mse_list, (alpha + (1.0 - alpha) / 2.0) * 100)
print(f"{alpha*100:.1f}% confidence interval for MSE: {lower:.4f} to {upper:.4f}")


# confidence intervals
lower = np.percentile(rmse_list, ((1.0 - alpha) / 2.0) * 100)
upper = np.percentile(rmse_list, (alpha + (1.0 - alpha) / 2.0) * 100)
print(f"{alpha*100:.1f}% confidence interval for RMSE: {lower:.4f} to {upper:.4f}")


# confidence intervals
lower = np.percentile(corr_list, ((1.0 - alpha) / 2.0) * 100)
upper = np.percentile(corr_list, (alpha + (1.0 - alpha) / 2.0) * 100)
print(f"{alpha*100:.1f}% confidence interval for CORR: {lower:.4f} to {upper:.4f}")


# confidence intervals
lower = np.percentile(r2_list, ((1.0 - alpha) / 2.0) * 100)
upper = np.percentile(r2_list, (alpha + (1.0 - alpha) / 2.0) * 100)
print(f"{alpha*100:.1f}% confidence interval for R2: {lower:.4f} to {upper:.4f}")
