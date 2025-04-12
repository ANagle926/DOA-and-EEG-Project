import keras
import numpy as np
from keras.src.utils.module_utils import scipy
from matplotlib import pyplot as plt
from joblib import dump, load
from VitalDBDataset import VitalDBDataset
from keras import Sequential
from keras.src.callbacks import ModelCheckpoint, EarlyStopping
from keras.src.layers import LSTM, Dense, Dropout, Bidirectional, GlobalAveragePooling1D, Flatten
from sklearn.metrics import mean_absolute_error, r2_score


dataset = VitalDBDataset(max_cases=5, srate=128)
dump(dataset, "Pre_processed_Data.joblib")
print("done saving dataset")
#dataset = load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/Data Files/dataset.joblib")

x_train, y_train = dataset.x_train, dataset.y_train
x_test, y_test = dataset.x_test, dataset.y_test
c_train, c_test = dataset.c_train, dataset.c_test
seglen=dataset.SEGLEN

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)


"""from scipy.fft import fft

# Take FFT of each wave
fft_features = np.abs(fft(x_train, axis=1))[:, :100, 0]  # First 100 frequency bins

# Use a simple model
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

X_train, X_val, y_train_small, y_val = train_test_split(fft_features, y_train, test_size=0.2, random_state=42)
model = Ridge()
model.fit(X_train, y_train_small)
print("Validation MAE:", np.mean(np.abs(model.predict(X_val) - y_val)))"""

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(seglen, 3)),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Bidirectional(LSTM(128, return_sequences=True)),
    GlobalAveragePooling1D(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dense(1)
])

model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
model.summary()

model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=15,
    batch_size=128,
    callbacks=[
        ModelCheckpoint('model.keras', save_best_only=True),
        EarlyStopping(patience=3, restore_best_weights=True)
    ]
)

dump(model, "/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/Data Files/Model Versions/LSTM_working_model_with_filter.joblib" )

test_loss, test_mae = model.evaluate(x_test, y_test)
print(f"Test Loss (MAE): {test_loss}")
print(f"Test MAE: {test_mae}")


#model.save(""LSTM_working_model.keras"")
#model= keras.models.load_model("../Data Files/Model Versions/LSTM_working_model.keras")

pred_test = model.predict(x_test).flatten()

for caseid in np.unique(c_test):
    case_mask = (c_test == caseid)
    pred_test[case_mask] = scipy.signal.medfilt(pred_test[case_mask], kernel_size=15)

# Calculate Mean Absolute Error
test_mae = mean_absolute_error(y_test, pred_test)
print(f'Test MAE: {test_mae}')

# Calculate correlation coefficient and R-squared
corr = np.corrcoef(y_test, pred_test)[0, 1]
r2 = r2_score(y_test, pred_test)
print(f'Correlation coefficient: {corr}')
print(f'R squared: {r2}')

# Scatter plot of actual vs. predicted DOA values
plt.figure(figsize=(6, 6))
plt.scatter(y_test, pred_test, s=1, alpha=0.5, color='violet')
plt.xlabel('Actual DOA')
plt.ylabel('Predicted DOA')
plt.title(f'Scatter Plot (Correlation: {corr:.4f})')
plt.plot([0, max(y_test)], [0, max(pred_test)], 'r--')
plt.show()

for caseid in np.random.choice(np.unique(c_test), size=3, replace=False):
    case_mask = (c_test == caseid)
    case_len = np.sum(case_mask)
    if case_len == 0:
        continue
    our_mae = np.mean(np.abs(y_test[case_mask] - pred_test[case_mask]))
    t = np.arange(case_len)
    plt.figure(figsize=(10, 4))
    plt.plot(t, y_test[case_mask], label='Actual DOA')
    plt.plot(t, pred_test[case_mask], label=f'Predicted DOA (DOA: {our_mae:.4f})')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('DOA')
    plt.title(f'Case {caseid}')
    plt.show()
    print(f'Case {caseid}, DOA: {our_mae:.4f}')

"""y_pred = model.predict(x_test).flatten()
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test MAE: {mae:.4f}")
print(f"R² Score: {r2:.4f}")

for caseid in np.random.choice(np.unique(c_test), size=3, replace=False):
    case_mask = (c_test == caseid)
    case_len = np.sum(case_mask)
    if case_len == 0:
        continue
    our_mae = np.mean(np.abs(y_test[case_mask] - y_pred[case_mask]))
    t = np.arange(case_len)
    plt.figure(figsize=(10, 4))
    plt.plot(t, y_test[case_mask], label='Actual DOA')
    plt.plot(t, y_pred[case_mask], label=f'Predicted DOA (DOA: {our_mae:.4f})')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('DOA')
    plt.title(f'Case {caseid}')
    plt.show()
    print(f'Case {caseid}, DOA: {our_mae:.4f}')"""

"""
Epoch 1/10
2143/2143 ━━━━━━━━━━━━━━━━━━━━ 139s 62ms/step - loss: 8.9004 - mean_absolute_error: 8.9004 - val_loss: 5.5210 - val_mean_absolute_error: 5.5210
Epoch 2/10
2143/2143 ━━━━━━━━━━━━━━━━━━━━ 130s 61ms/step - loss: 5.8208 - mean_absolute_error: 5.8208 - val_loss: 5.4419 - val_mean_absolute_error: 5.4419
Epoch 3/10
2143/2143 ━━━━━━━━━━━━━━━━━━━━ 131s 61ms/step - loss: 4.9443 - mean_absolute_error: 4.9443 - val_loss: 5.1938 - val_mean_absolute_error: 5.1938
Epoch 4/10
2143/2143 ━━━━━━━━━━━━━━━━━━━━ 131s 61ms/step - loss: 4.7931 - mean_absolute_error: 4.7931 - val_loss: 4.9351 - val_mean_absolute_error: 4.9351
Epoch 5/10
2143/2143 ━━━━━━━━━━━━━━━━━━━━ 131s 61ms/step - loss: 4.5933 - mean_absolute_error: 4.5933 - val_loss: 4.7890 - val_mean_absolute_error: 4.7890
Epoch 6/10
2143/2143 ━━━━━━━━━━━━━━━━━━━━ 131s 61ms/step - loss: 4.5229 - mean_absolute_error: 4.5229 - val_loss: 5.1322 - val_mean_absolute_error: 5.1322
Epoch 7/10
2143/2143 ━━━━━━━━━━━━━━━━━━━━ 131s 61ms/step - loss: 4.4353 - mean_absolute_error: 4.4353 - val_loss: 4.3865 - val_mean_absolute_error: 4.3865
Epoch 8/10
2143/2143 ━━━━━━━━━━━━━━━━━━━━ 131s 61ms/step - loss: 4.3507 - mean_absolute_error: 4.3507 - val_loss: 4.5845 - val_mean_absolute_error: 4.5845
Epoch 9/10
2143/2143 ━━━━━━━━━━━━━━━━━━━━ 131s 61ms/step - loss: 4.3004 - mean_absolute_error: 4.3004 - val_loss: 4.5673 - val_mean_absolute_error: 4.5673
Epoch 10/10
2143/2143 ━━━━━━━━━━━━━━━━━━━━ 132s 62ms/step - loss: 4.4357 - mean_absolute_error: 4.4357 - val_loss: 4.7963 - val_mean_absolute_error: 4.7963
5163/5163 ━━━━━━━━━━━━━━━━━━━━ 103s 20ms/step - loss: 4.5068 - mean_absolute_error: 4.5068
Test accuracy: 4.386460304260254
Test MAE: 4.386460304260254


Test MAE: 3.6183362532226817
Correlation coefficient: 0.8903530133411133
R squared: 0.7844026511838958



Rerun:


Epoch 1/10
I0000 00:00:1739936883.011491  576052 cuda_dnn.cc:529] Loaded cuDNN version 90300
2143/2143 ━━━━━━━━━━━━━━━━━━━━ 0s 195ms/step - loss: 8.6729 - mean_absolute_error: 8.67292025-02-18 22:55:02.691423: W external/local_xla/xla/tsl/framework/cpu_allocator_impl.cc:83] Allocation of 338356224 exceeds 10% of free system memory.
2143/2143 ━━━━━━━━━━━━━━━━━━━━ 475s 219ms/step - loss: 8.6721 - mean_absolute_error: 8.6721 - val_loss: 5.1842 - val_mean_absolute_error: 5.1842
Epoch 2/10
2143/2143 ━━━━━━━━━━━━━━━━━━━━ 469s 219ms/step - loss: 5.8391 - mean_absolute_error: 5.8391 - val_loss: 5.9554 - val_mean_absolute_error: 5.9554
Epoch 3/10
2143/2143 ━━━━━━━━━━━━━━━━━━━━ 468s 218ms/step - loss: 5.0480 - mean_absolute_error: 5.0480 - val_loss: 5.0037 - val_mean_absolute_error: 5.0037
Epoch 4/10
2143/2143 ━━━━━━━━━━━━━━━━━━━━ 468s 218ms/step - loss: 4.7324 - mean_absolute_error: 4.7324 - val_loss: 4.6516 - val_mean_absolute_error: 4.6516
Epoch 5/10
2143/2143 ━━━━━━━━━━━━━━━━━━━━ 464s 216ms/step - loss: 4.5726 - mean_absolute_error: 4.5726 - val_loss: 4.7689 - val_mean_absolute_error: 4.7689
Epoch 6/10
2143/2143 ━━━━━━━━━━━━━━━━━━━━ 468s 218ms/step - loss: 4.4563 - mean_absolute_error: 4.4563 - val_loss: 4.6341 - val_mean_absolute_error: 4.6341
Epoch 7/10
2143/2143 ━━━━━━━━━━━━━━━━━━━━ 473s 221ms/step - loss: 4.3673 - mean_absolute_error: 4.3673 - val_loss: 4.9845 - val_mean_absolute_error: 4.9845
Epoch 8/10
2143/2143 ━━━━━━━━━━━━━━━━━━━━ 463s 216ms/step - loss: 4.3190 - mean_absolute_error: 4.3190 - val_loss: 4.5144 - val_mean_absolute_error: 4.5144
Epoch 9/10
2143/2143 ━━━━━━━━━━━━━━━━━━━━ 469s 219ms/step - loss: 4.2548 - mean_absolute_error: 4.2548 - val_loss: 4.7987 - val_mean_absolute_error: 4.7987
Epoch 10/10
2143/2143 ━━━━━━━━━━━━━━━━━━━━ 466s 217ms/step - loss: 4.2113 - mean_absolute_error: 4.2113 - val_loss: 4.5361 - val_mean_absolute_error: 4.5361
2025-02-19 00:06:01.096852: W external/local_xla/xla/tsl/framework/cpu_allocator_impl.cc:83] Allocation of 338356224 exceeds 10% of free system memory.
5163/5163 ━━━━━━━━━━━━━━━━━━━━ 318s 62ms/step
Test MAE: 4.5144
R² Score: 0.6624




"""



