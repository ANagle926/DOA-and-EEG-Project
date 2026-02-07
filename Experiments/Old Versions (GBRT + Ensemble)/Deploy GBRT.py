import numpy as np
from joblib import dump, load
from keras import Sequential, Model
from keras.src.saving import register_keras_serializable, load_model
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import os
from keras.src.layers import Input, Dense, Dropout, Conv1D, Bidirectional, LayerNormalization, LSTM, MaxPooling1D, GlobalAveragePooling1D, MultiHeadAttention
from keras.src.optimizers import Adam
from keras.src.callbacks import EarlyStopping
import keras
from keras.src.optimizers.schedules import CosineDecayRestarts
import tensorflow as tf
from keras import layers
from sklearn.ensemble import GradientBoostingRegressor
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

def create_ensemble(x_train, y_train, n_models=5):

    x_train = x_train.reshape((-1, 1024, 1))
    inputs = Input(shape=(1024, 1))

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
            loss='huber',
            metrics=['mae']
        )

        model.fit(
            x_train, y_train,
            epochs=15,
            batch_size=batch_size,
            verbose=1
        )
        model.save(f'ensemble_model_raw_v1.{i}.keras')

def create_GBRT(x_train, x_test, y_train, n_models=5):
    train_preds = []
    test_preds = []

    #STILL WRONG:

    for i in range(n_models):
        model = load_model(f'ensemble_model_raw_v1.{i}.keras')
        train_preds.append(model.predict(x_train, verbose=0))
        test_preds.append(model.predict(x_test, verbose=0))

    P_test  = np.hstack(test_preds)
    P_train = np.hstack(train_preds)

    y_train = y_train.flatten()

    gbrt = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )
    gbrt.fit(P_train, y_train)

    dump(gbrt, "../../Files/Saved Model Files/GBRT/gbrt_model_150_raw_new.pkl")
    #gbrt= load("Files/Saved Model Files/GBRT/gbrt_model_150_raw_new.pkl")

    final_predictions = gbrt.predict(P_test)

    return final_predictions

def deploy_GBRT (x_test, n_models=5):
    test_preds = []

    for i in range(n_models):
        model = load_model(f'Files/Saved Model Files/Ensemble/ensemble_model_150_raw.{i}.keras')
        test_preds.append(model.predict(x_test, verbose=0))

    P_test  = np.hstack(test_preds)
    gbrt= load("../../Files/Saved Model Files/GBRT/gbrt_model_150_raw_new.pkl")

    final_predictions = gbrt.predict(P_test)

    return final_predictions

dataset=VitalDBDataset(max_cases=150)
x_train = dataset.x_train
x_test = dataset.x_test
y_train = dataset.y_train
y_test = dataset.y_test

n_models=5
create_ensemble(x_train, y_train, n_models=n_models)
preds= create_GBRT(x_train, x_test, y_train)

#meta_test_preds = deploy_GBRT(x_test)

mae = mean_absolute_error(y_test, preds)
mse = mean_squared_error(y_test, preds)
corr = np.corrcoef(y_test, preds)[0, 1]
r2 = r2_score(y_test, preds)

print(f"🔍 Meta-Ensemble GBRT MAE on Test Set: {mae:.4f}")
print(f"🔍 Meta-Ensemble GBRT MSE on Test Set: {mse:.4f}")
print(f"🔍 Meta-Ensemble GBRT CORR on Test Set: {corr:.4f}")
print(f"🔍 Meta-Ensemble GBRT RSQUARED on Test Set: {r2:.4f}")
