import io
from fastapi import FastAPI, UploadFile, File, HTTPException
import scipy.io
import mne
import numpy as np
from keras import Sequential
from keras.src.saving import register_keras_serializable, load_model
from keras.src.layers import Dense, Dropout,  LayerNormalization, MultiHeadAttention
import tensorflow as tf
from keras import layers
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
app = FastAPI(title="EEG → BIS API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.options("/predict")
def predict_options():
    return Response(status_code=200)

@register_keras_serializable(package="Custom")
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
@register_keras_serializable(package="Custom")
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


MODEL_PATH = "raw_model_v3.keras"
model = load_model(MODEL_PATH, custom_objects={"PositionalEmbedding": PositionalEmbedding, "TransformerBlock": TransformerBlock})

S_RATE = 128
SEG_LEN = 1024  # 128 Hz * 8 sec

def _segment_1d_signal(sig: np.ndarray) -> np.ndarray:
    """Convert 1D signal into (n,1024)"""
    sig = sig.astype(np.float32)
    n = sig.size // SEG_LEN
    if n == 0:
        raise HTTPException(status_code=400, detail="Not enough samples for one 8-second segment.")
    return sig[: n * SEG_LEN].reshape(n, SEG_LEN)


def _load_eeg_file_to_2d_array(upload: UploadFile) -> np.ndarray:
    filename = (upload.filename or "").lower()
    raw_bytes = upload.file.read()

    if filename.endswith(".npy"):
        arr = np.load(io.BytesIO(raw_bytes), allow_pickle=False)

    elif filename.endswith(".csv") or filename.endswith(".txt"):
        text = raw_bytes.decode("utf-8", errors="ignore")
        try:
            arr = np.loadtxt(io.StringIO(text), delimiter=",")
        except Exception:
            arr = np.loadtxt(io.StringIO(text))

    elif filename.endswith(".mat"):
        mat = scipy.io.loadmat(io.BytesIO(raw_bytes))
        candidates = [
            v for v in mat.values()
            if isinstance(v, np.ndarray) and v.size >= SEG_LEN
        ]
        if not candidates:
            raise HTTPException(status_code=400, detail="No valid EEG array found in .mat file.")
        arr = candidates[0].squeeze()

    elif filename.endswith(".edf"):
        with io.BytesIO(raw_bytes) as bio:
            raw = mne.io.read_raw_edf(bio, preload=True, verbose=False)

        if int(raw.info["sfreq"]) != S_RATE:
            raise HTTPException(
                status_code=400,
                detail=f"EDF sampling rate must be {S_RATE} Hz."
            )

        data = raw.get_data()  # shape: (channels, samples)

        # Use first channel by default
        arr = data[0]

    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Upload .npy, .csv, .txt, .mat, or .edf."
        )

    arr = np.array(arr, dtype=np.float32)

    # Shape normalization
    """if arr.ndim == 1:
        arr = _segment_1d_signal(arr)

    elif arr.ndim == 2:
        if arr.shape[1] == SEG_LEN:
            pass
        elif arr.shape[0] == SEG_LEN:
            arr = arr.T
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Expected shape (n,1024). Got {arr.shape}."
            )
    else:
        raise HTTPException(status_code=400, detail="EEG array must be 1D or 2D.")"""

    return arr


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    x_test = _load_eeg_file_to_2d_array(file)

    preds = model.predict(x_test, verbose=0).flatten()
    preds = np.clip(preds, 0, 100)

    return {"predictions": preds.tolist()}
