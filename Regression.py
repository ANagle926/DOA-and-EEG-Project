
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV



def create_model(units=64, dropout=0.3):
    model = Sequential()
    model.add(LSTM(units, return_sequences=True, input_shape=(SEGLEN, 1)))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Bidirectional(LSTM(units, return_sequences=True)))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(dropout))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
    return model

# Define KerasRegressor
model = KerasRegressor(
    model=create_model,  # Use create_model function
    verbose=1
)
param_grid = {
    'model__units': [64, 100],
    'model__dropout': [0.6, 0.7],
    'batch_size': [150, 200, 250],
    'epochs': [10, 15]
}

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=2)
grid_result = grid.fit(x_train, y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

import tensorflow.keras.models.Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, GlobalAveragePooling1D
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential



def create_model(units=64, dropout=0.3):
    model = Sequential()
    model.add(LSTM(units, return_sequences=True, input_shape=(SEGLEN, 1)))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Bidirectional(LSTM(units, return_sequences=True)))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(dropout))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
    return model


# Define KerasRegressor
model = KerasRegressor(
    model=create_model,  # Use create_model function
    verbose=1
)
param_grid = {
    'model__units': [64, 100],
    'model__dropout': [0.6, 0.7],
    'batch_size': [150, 200, 250],
    'epochs': [10, 15]
}

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=2)
grid_result = grid.fit(x_train, y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))



import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

def analyze_model(model, x_test, y_test, c_test):
    # Make predictions on the test set
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
    plt.plot([0, 2], [0, 2], 'r--')
    plt.xlim([0, 2])
    plt.ylim([0, 2])
    plt.show()

      # Plot predictions for random test cases
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

#LSTM--- works (tested)

from tensorflow import keras
from tensorflow.keras.layers import Input, RNN, LSTMCell, Dense, Dropout
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional, LSTM

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

test_mae, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_accuracy}")
print(f"Test MAE: {test_mae}")

analyze_model(model, x_test, y_test, c_test)

test_model = Sequential()
test_model.add(LSTM(64,return_sequences=True))
test_model.add(Dense(64, activation='relu'))
test_model.add(Dropout(0.5))
test_model.add(Bidirectional(LSTM(128, return_sequences=True)))
test_model.add(GlobalAveragePooling1D())
test_model.add(Dropout(0.7))
test_model.add(Dense(64, activation='relu'))
test_model.add(Dropout(0.5))
test_model.add(Dense(64, activation='relu'))
test_model.add(Dense(1))
test_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

test_model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=15,
    batch_size=256,
    callbacks=[
        ModelCheckpoint('model.keras', save_best_only=True),
        EarlyStopping(patience=3, restore_best_weights=True)
    ]
)

test_mae, test_accuracy = test_model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_accuracy}")
print(f"Test MAE: {test_mae}")
print(f"    ")
print(f"    ")
analyze_model(test_model, x_test, y_test, c_test)