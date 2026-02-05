from keras.src.saving import load_model
import io
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
import scipy.io
import mne

app = FastAPI(title="EEG → BIS API")

# -----------------------
# Load model ONCE
# -----------------------
MODEL_PATH = "raw_model_v3.keras"
model = load_model(MODEL_PATH)

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
    if arr.ndim == 1:
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
        raise HTTPException(status_code=400, detail="EEG array must be 1D or 2D.")

    return arr


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    x_test = _load_eeg_file_to_2d_array(file)

    preds = model.predict(x_test, verbose=0).flatten()
    preds = np.clip(preds, 0, 100)

    return {"predictions": preds.tolist()}
