import numpy as np
from matplotlib import pyplot as plt
from joblib import dump, load
from Dataset2 import Dataset2
from keras import Sequential
from keras.src.callbacks import ModelCheckpoint, EarlyStopping
from keras.src.layers import LSTM, Dense, Dropout, Bidirectional, GlobalAveragePooling1D

from sklearn.metrics import mean_absolute_error, r2_score

dataset = Dataset2(max_cases=100, srate=128)
dump(dataset, "LSTM_data.joblib")
#dataset = load("my_object.joblib")


x_train, y_train = dataset.x_train, dataset.y_train
x_test, y_test = dataset.x_test, dataset.y_test
c_train, c_test = dataset.c_train, dataset.c_test
seglen = dataset.SEGLEN

model = Sequential()
model.add(LSTM(64,return_sequences=True))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(GlobalAveragePooling1D())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(64))
model.add(Dense(1))
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
print("here")
model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=10,
    batch_size=256,
    callbacks=[
        ModelCheckpoint('model.keras', save_best_only=True),
        EarlyStopping(patience=3, restore_best_weights=True)
    ]
)
"""
test_loss, test_mae = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_loss}")
print(f"Test MAE: {test_mae}")
"""

model.save("""LSTM_working_model.keras""")

y_pred = model.predict(x_test).flatten()
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test MAE: {mae:.4f}")
print(f"R² Score: {r2:.4f}")

for caseid in np.random.choice(np.unique(c_test), size=3, replace=False):
    case_mask = (c_test == caseid)
    case_len = np.sum(case_mask)
    if case_len == 0:
        continue
    print("y_test shape", y_test.shape)
    print("c_test shape", c_test.shape)
    print("case_mask", case_mask)
    print("case_len", case_len)
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
    print(f'Case {caseid}, DOA: {our_mae:.4f}')

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

"""



