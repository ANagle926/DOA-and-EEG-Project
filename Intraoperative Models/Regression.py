import keras
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from imblearn.tensorflow.tests.test_generator import tf

from keras import Sequential
from keras.src.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.src.layers import LSTM, Dense, Dropout, Bidirectional, GlobalAveragePooling1D, LayerNormalization, \
    GlobalMaxPooling1D
from keras.src.optimizers import Adam

from scikeras.wrappers import KerasRegressor
from scipy.signal import medfilt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.python.keras.regularizers import l2


class ValidationLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs:
            print(f"Epoch {epoch+1}: val_mae = {logs.get('val_mean_absolute_error', 'Not Available')}")


class EEGRegressor:
    def __init__(self, x_train, y_train, x_test, y_test, c_test, seglen):
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        self.c_test=c_test

        self.seglen = seglen  # Segment length (SEGLEN)

        # Creating variables
        self.x_train_resampled = None
        self.x_test_resampled = None
        self.y_train_resampled = None
        self.y_test_resampled = None
        self.model = None

        self.preprocess_data()

    def preprocess_data(self):

        self.x_train_resampled = self.x_train.reshape(-1, self.seglen, 1)
        self.x_test_resampled = self.x_test.reshape(-1, self.seglen, 1)

    def create_model(self, units=64, dropout=0.5, reg_strength=0.001, learning_rate=0.001):

        model = Sequential([
            LSTM(units, return_sequences=True, input_shape=(self.seglen, 1)),
            Dense(64, activation='relu'),
            Dropout(dropout),
            Bidirectional(LSTM(64, return_sequences=True, input_shape=(self.seglen, 1), kernel_regularizer=keras.regularizers.l2(reg_strength))),
            GlobalAveragePooling1D(),
            Dropout(dropout),
            Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(reg_strength)),
            Dropout(dropout),
            Dense(64, activation='relu'),
            Dense(1)
        ])
        """
        model = Sequential([
            Bidirectional(LSTM(units, return_sequences=True, input_shape=(self.seglen, 1))),
            LayerNormalization(),  # Stabilizes LSTM output
            LSTM(32, return_sequences=True),  # Extra LSTM for better feature extraction
            Dense(64, activation=tf.nn.swish),
            LayerNormalization(),
            Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=keras.regularizers.l2(reg_strength))),
            GlobalMaxPooling1D(),  # Reduces variance in pooling
            Dense(64, activation=tf.nn.swish, kernel_regularizer=keras.regularizers.l2(reg_strength)),  # Reduced Dense size
            Dropout(dropout),
            Dense(64, activation=tf.nn.swish),
            Dense(1)
        ])"""


        optimizer = Adam(learning_rate=learning_rate)  # Use learning_rate from GridSearch
        model.compile(loss='mae', optimizer=optimizer, metrics=['mae'])
        return model

    def train_model(self, epochs=15, batch_size=256):

        self.model = self.create_model()

        # Train the model with checkpointing
        self.model.fit(
            self.x_train_resampled, self.y_train,
            validation_data=(self.x_test_resampled, self.y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[ModelCheckpoint('../Data Files/Model Versions/model.keras', save_best_only=True),
                       EarlyStopping(patience=3, restore_best_weights=True)]
        )

    def hyperparameter_tuning(self):

        model = KerasRegressor(model=lambda units, dropout, reg_strength, learning_rate: self.create_model(units=units, dropout=dropout, reg_strength=reg_strength, learning_rate=learning_rate), verbose=1, callbacks=[ValidationLogger()])

        print("KerasRegressor initialized.")

        param_grid = {
            'model__units': [90],
            'model__dropout': [0.1, 0.15],
            'model__reg_strength': [0.0005],
            'model__learning_rate': [0.0005],
            'batch_size': [128],
            'epochs': [15]
        }

        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=2, return_train_score=True, error_score='raise')
        print("GridSearchCV initialized.")
        grid_result = grid.fit(
            self.x_train_resampled, self.y_train,
            **{
            "validation_data": (self.x_test_resampled, self.y_test),
            "callbacks": [ValidationLogger()]  # Ensures val_mae is printed
        })



        if hasattr(grid_result, 'best_score_') and hasattr(grid_result, 'best_params_'):
            print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")
            return grid_result.best_params_
        else:
            print("Grid search did not return best parameters.")
            return None


    def evaluate_model(self):

        """pred_test = self.model.predict(self.x_test_resampled).flatten()

        # Calculate Mean Absolute Error and R-squared
        test_mae = mean_absolute_error(self.y_test, pred_test)
        r2 = r2_score(self.y_test, pred_test)
        print(f'Test MAE: {test_mae}')
        print(f'R squared: {r2}')


        # Scatter plot of actual vs predicted values
        plt.figure(figsize=(6, 6))
        plt.scatter(self.y_test, pred_test, s=1, alpha=0.5, color='violet')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Scatter Plot (R2: {r2:.4f})')
        plt.plot([0, max(self.y_test)], [0, max(pred_test)], 'r--')
        plt.show()"""

        pred_test = self.model.predict(self.x_test_resampled).flatten()

        for caseid in np.unique(self.c_test):
            case_mask = (self.c_test == caseid)
            pred_test[case_mask] = scipy.signal.medfilt(pred_test[case_mask], kernel_size=15)

        # Calculate Mean Absolute Error
        test_mae = mean_absolute_error(self.y_test, pred_test)
        print(f'Test MAE: {test_mae}')

        # Calculate correlation coefficient and R-squared
        corr = np.corrcoef(self.y_test, pred_test)[0, 1]
        r2 = r2_score(self.y_test, pred_test)
        print(f'Correlation coefficient: {corr}')
        print(f'R squared: {r2}')

        # Scatter plot of actual vs. predicted DOA values
        plt.figure(figsize=(6, 6))
        plt.scatter(self.y_test, pred_test, s=1, alpha=0.5, color='violet')
        plt.xlabel('Actual DOA')
        plt.ylabel('Predicted DOA')
        plt.title(f'Scatter Plot (Correlation: {corr:.4f})')
        plt.plot([0, max(self.y_test)], [0, max(pred_test)], 'r--')
        plt.show()

        for caseid in np.random.choice(np.unique(self.c_test), size=3, replace=False):
            case_mask = (self.c_test == caseid)
            case_len = np.sum(case_mask)
            if case_len == 0:
                continue
            our_mae = np.mean(np.abs(self.y_test[case_mask] - pred_test[case_mask]))
            t = np.arange(case_len)
            plt.figure(figsize=(10, 4))
            plt.plot(t, self.y_test[case_mask], label='Actual DOA')
            plt.plot(t, pred_test[case_mask], label=f'Predicted DOA (DOA: {our_mae:.4f})')
            plt.legend()
            plt.xlabel('Time')
            plt.ylabel('DOA')
            plt.title(f'Case {caseid}')
            plt.show()
            print(f'Case {caseid}, DOA: {our_mae:.4f}')

        return test_mae, r2
