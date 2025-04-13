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


""""
To Do:

save this version and create another file for hyperparameter tuning => maybe use regression file?
(current Test MAE: 3.9506
Correlation coefficient: 0.8545
R squared: 0.5980)

maybe dont evaluate based on r^2 because the residual variation is low => 
most EEG waves are from normal patients, so there isnt variation in x=>
there is clustering of x-values so outliers have leverage

work on AI filter: make it more accurate or delete model
=> evaluate correlation with y based on alpha, beta, and gamma wave (see EEMD tuning)

**MAKE SURE TO SAVE GRAPHS AND Model iterations

currently working with 1 case ID => will need to expand to improve r^2 and general accuracy

"""




#dataset = VitalDBDataset(max_cases=1, srate=128)
#dump(dataset, "Processed_Data.joblib")
#print("done saving dataset")
dataset=load("Processed_Data.joblib")


x_train, y_train = dataset.x_train, dataset.y_train
x_test, y_test = dataset.x_test, dataset.y_test
c_train, c_test = dataset.c_train, dataset.c_test
seglen=dataset.SEGLEN

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)


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

dump(model, "/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/Data Files/Model Versions/LSTM_working_model_with_filter.joblib")

#model=load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/Data Files/Model Versions/LSTM_working_model_with_filter.joblib")


# Predict and evaluate test statistics
pred_test = model.predict(x_test).flatten()
test_mae = mean_absolute_error(y_test, pred_test)
corr = np.corrcoef(y_test, pred_test)[0, 1]
r2 = r2_score(y_test, pred_test)
print(f"Test MAE: {test_mae:.4f}")
print(f"Correlation coefficient: {corr:.4f}")
print(f"R squared: {r2:.4f}")


""""# 1. Scatter plot: Actual vs Predicted
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

# 4. Time-series plots with moving average overlay (3 random cases)
def moving_avg(signal, window=15):
    return np.convolve(signal, np.ones(window)/window, mode='same')

for caseid in np.random.choice(np.unique(c_test), size=1, replace=False):
    case_mask = (c_test == caseid)
    case_len = np.sum(case_mask)
    if case_len == 0:
        continue

    actual = y_test[case_mask]
    predicted = pred_test[case_mask]
    t = np.arange(case_len)
    mae = np.mean(np.abs(actual - predicted))

    plt.figure(figsize=(12, 4))
    plt.plot(t, actual, label='Actual BIS', color='black')
    plt.plot(t, predicted, label='Predicted BIS', color='royalblue', alpha=0.7)
    plt.plot(t, moving_avg(predicted), label='Predicted (Moving Avg)', linestyle='--', color='orange')
    plt.xlabel('Time')
    plt.ylabel('BIS')
    plt.title(f'Case {caseid} | MAE: {mae:.4f}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()"""



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



