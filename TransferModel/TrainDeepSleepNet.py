import os
import numpy as np
import mne
from sklearn.model_selection import train_test_split
from keras import layers, models

def load_sleep_edf_record(eeg_path, hypnogram_path):
    # Load and preprocess EEG
    raw = mne.io.read_raw_edf(eeg_path, preload=True)
    raw.pick(picks=['EEG Fpz-Cz'])
    raw.resample(100)

    # Process annotations
    annotations = mne.read_annotations(hypnogram_path)
    raw.set_annotations(annotations)

    # Stage mapping with merged deep sleep
    mapping = {
        'Sleep stage W': 0,
        'Sleep stage 1': 1,
        'Sleep stage 2': 2,
        'Sleep stage 3': 3,  # Combined stage 3/4
        'Sleep stage 4': 3,
        'Sleep stage R': 4,
        'Sleep stage ?': -1,
        'Movement time': -1,
    }

    # Create events from annotations
    events, event_id = mne.events_from_annotations(raw, event_id=mapping)

    # Create epochs with critical fixes
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=0,
        tmax=29.99,
        baseline=None,
        detrend=0,  # Fix 1: Disable detrending to prevent empty epoch errors
        picks=['EEG Fpz-Cz'],
        preload=True,
        on_missing='ignore'  # Fix 2: Handle missing events gracefully
    )

    # Get data and labels
    x = epochs.get_data()  # (n_epochs, 1, 3000)
    y = epochs.events[:, 2]

    # Filter out unknown/movement stages
    valid_idx = y != -1  # Fix 3: Remove invalid labels
    return x[valid_idx], y[valid_idx]

def build_deepsleepnet_keras(sequence_length):
    # Model architecture remains the same
    inputs = layers.Input(shape=(sequence_length, 1))
    x = layers.Conv1D(64, 5, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(128, 5, padding='same', activation='relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(256, 5, padding='same', activation='relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(5, activation='softmax')(x)
    return models.Model(inputs, x)

# Enhanced data loading with debugging
# Corrected file loading logic
data_folder = "/home/anika/sleep-edf/sleep-cassette"
all_x, all_y = [], []

for file in os.listdir(data_folder):
    if file.endswith("-PSG.edf"):
        base_name = file.split("-PSG.edf")[0]
        eeg_file = os.path.join(data_folder, file)
        hypnogram_file = os.path.join(data_folder, f"{base_name}-Hypnogram.edf")

        # Check for alternative hypnogram naming
        if not os.path.exists(hypnogram_file):
            hypnogram_file = os.path.join(data_folder, f"{base_name.replace('E0','EC')}-Hypnogram.edf")

        if os.path.exists(hypnogram_file):
            try:
                x, y = load_sleep_edf_record(eeg_file, hypnogram_file)
                all_x.append(x)
                all_y.append(y)
                print(f"✅ Successfully processed {base_name}")
            except Exception as e:
                print(f"❌ Error processing {base_name}: {str(e)}")
        else:
            print(f"⚠️ Missing hypnogram for {base_name}")

# Handle empty case
if not all_x:
    raise ValueError("No valid data files found!")

# Combine data
x_all = np.concatenate(all_x, axis=0)
y_all = np.concatenate(all_y, axis=0)
print(f"\n✅ Final dataset: {x_all.shape[0]} samples")

# Train-test split
x_train, x_val, y_train, y_val = train_test_split(
    x_all, y_all,
    test_size=0.2,
    random_state=42,
    stratify=y_all
)

x_train = x_train.transpose(0, 2, 1)  # (samples, 3000, 1)
x_val = x_val.transpose(0, 2, 1)

# Build and train model
model = build_deepsleepnet_keras(3000)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=20,
    batch_size=128,
    verbose=1
)

# Save model
model.save_weights("deepsleepnet_sleepedf_weights.h5")
print("Training complete!")

