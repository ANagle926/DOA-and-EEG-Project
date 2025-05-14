import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR
import warnings
import sys
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
sys.stderr = open(os.devnull, 'w')  # Redirects all stderr output

import keras
import numpy as np
import torch
from keras import Sequential
from keras.src.optimizers import Adam
from keras.src.regularizers import regularizers
from keras.src.saving import load_model
from sklearn.metrics import mean_absolute_error
from EEGPT.downstream.Modules.models.EEGPT_mcae_finetune import EEGPTClassifier
import numpy as np
from joblib import dump, load
from keras.src.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.src.layers import MaxPooling1D, Bidirectional, LayerNormalization, LSTM, GlobalAveragePooling1D, Dense, \
    Dropout, Conv1D
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

from torch import amp
@amp.autocast("cuda")


def forward_features_only(self, x):
    return self.forward_features(x, return_patch_tokens=True, return_all_tokens=False)

def plot_data(x_train, x_test, y_train, y_test):
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

def load_transfer_model():
    # Load checkpoint
    ckpt_path = "/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/EEGPT/checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt"

    checkpoint = torch.load(ckpt_path, map_location='cpu')

    # Create model
    model = EEGPTClassifier(
        eeg_size=(3, 1024),
        patch_size=64,
        emb_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        num_classes=2,
        use_channels_names=['Fp1', 'Fp2', 'F3'],
        desired_time_len=1024
    )

    # Load weights
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()

    # Freeze weights
    for param in model.parameters():
        param.requires_grad = False

    return model

def find_features(x_train, x_test, y_train, y_test):

    model = load_transfer_model()

    x_train_reshaped = np.transpose(x_train, (0, 2, 1))  # (samples, 3, 1024)
    x_test_reshaped = np.transpose(x_test, (0, 2, 1))
    x_train_tensor = torch.tensor(x_train_reshaped, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test_reshaped, dtype=torch.float32)

    model.forward_features_only = forward_features_only.__get__(model)
    batch_size = 64

    train_features = []
    test_features = []

    print("🔄 Extracting features in batches...")
    model.eval()
    with torch.no_grad():
        for i in range(0, len(x_train_tensor), batch_size):
            batch = x_train_tensor[i:i+batch_size]
            batch_features = model.forward_features_only(batch).cpu().numpy()
            train_features.append(batch_features)

    model.forward_features_only = forward_features_only.__get__(model)

    with torch.no_grad():
        for i in range(0, len(x_test_tensor), batch_size):
            batch = x_test_tensor[i:i+batch_size]
            batch_features = model.forward_features_only(batch).cpu().numpy()
            test_features.append(batch_features)

    train_features = np.concatenate(train_features, axis=0)
    test_features = np.concatenate(test_features, axis=0)

    np.save("eegpt_train_features.npy", train_features)
    np.save("eegpt_train_labels.npy", y_train)
    np.save("eegpt_test_features.npy", test_features)
    np.save("eegpt_test_labels.npy", y_test)

    print(f"✅ Saved features: eegpt_train_features.npy  shape: {train_features.shape}")
    print(f"✅ Saved labels: eegpt_train_labels.npy      shape: {y_train.shape}")

def moving_avg(signal, window=15):
    return np.convolve(signal, np.ones(window)/window, mode='same')

def evaluate_model(model, x_test, y_test):

    pred_test = model.predict(x_test).flatten()

    test_mae = mean_absolute_error(y_test, pred_test)
    corr = np.corrcoef(y_test, pred_test)[0, 1]
    r2 = r2_score(y_test, pred_test)
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Correlation coefficient: {corr:.4f}")
    print(f"R squared: {r2:.4f}")

    # 1. Scatter plot: Actual vs Predicted
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, pred_test, s=1, alpha=0.5, color='violet')
    plt.xlabel('Actual BIS')
    plt.ylabel('Predicted BIS')
    plt.title(f'Scatter Plot (Correlation: {corr:.4f})')
    plt.plot([0, max(y_test)], [0, max(y_test)], 'r--')
    plt.grid(True)
    plt.show()

    # 2. Histogram of prediction errors
    errors = y_test - pred_test
    plt.figure(figsize=(8, 4))
    plt.hist(errors, bins=50, color='steelblue', edgecolor='black')
    plt.xlabel('Prediction Error (Actual - Predicted)')
    plt.ylabel('Count')
    plt.title('Histogram of Prediction Errors')
    plt.grid(True)
    plt.show()

    # 3. Scatter plot colored by absolute error
    abs_errors = np.abs(errors)
    plt.figure(figsize=(6, 6))
    sc = plt.scatter(y_test, pred_test, c=abs_errors, s=2, cmap='viridis', alpha=0.6)
    plt.xlabel('Actual BIS')
    plt.ylabel('Predicted BIS')
    plt.title('Scatter Plot Colored by Absolute Error')
    plt.colorbar(sc, label='Absolute Error')
    plt.plot([0, max(y_test)], [0, max(y_test)], 'r--')
    plt.grid(True)
    plt.show()

def create_bis_regressor_model(x_train, y_train, x_test, y_test):

    num_patches = x_train.shape[1]
    embedding_dim = x_train.shape[2]
    #6.42

    model = Sequential([
        Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(num_patches, embedding_dim)),
        Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0005),input_shape=(num_patches, embedding_dim)),
        MaxPooling1D(pool_size=2),
        Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0005)),
        Bidirectional(LSTM(256, return_sequences=True, kernel_regularizer=keras.regularizers.l2(0.0005))),
        LayerNormalization(),
        LSTM(128, return_sequences=True, kernel_regularizer=keras.regularizers.l2(0.0005)),
        GlobalAveragePooling1D(),
        LayerNormalization(),
        Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0005)),
        Dropout(0.1),
        Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0005)),
        Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0005)),
        Dense(1)

    ])

    model.compile(optimizer=Adam(learning_rate=1e-4), loss='mse', metrics=['mae'])

    reduce_lr = ReduceLROnPlateau(
        monitor='val_mae',
        factor=0.5,            # Reduce LR by a factor of 0.5
        patience=5,            # Wait 4 epochs with no improvement
        min_lr=1e-6,           # Don't go below this learning rate
        verbose=1              # Print when LR is reduced
    )
    early_stop = EarlyStopping(
        monitor='val_mae',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=80,
        batch_size=128,
        callbacks=[
            early_stop,
            reduce_lr
        ],
        verbose=1
    )

    model.save("eeg_regressor.keras")

"""dataset=load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/Data Files/dataset_twenty_cases_SEGLENMID.joblib")

x_train, y_train = dataset.x_train, dataset.y_train
x_test, y_test = dataset.x_test, dataset.y_test
print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

plot_data(x_train, x_test, y_train, y_test)

find_features(x_train, x_test, y_train, y_test)"""

x_train = np.load("eegpt_train_features.npy")
y_train = np.load("eegpt_train_labels.npy")
x_test = np.load("eegpt_test_features.npy")
y_test = np.load("eegpt_test_labels.npy")

print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

plot_data(x_train, x_test, y_train, y_test)

create_bis_regressor_model(x_train, y_train, x_test, y_test)

model = load_model("eeg_regressor.keras")

evaluate_model(model, x_test, y_test)
