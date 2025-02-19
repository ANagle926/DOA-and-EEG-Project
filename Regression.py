import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

from keras import Sequential
from keras.src.callbacks import ModelCheckpoint, EarlyStopping
from keras.src.layers import LSTM, Dense, Dropout, Bidirectional, GlobalAveragePooling1D

from scikeras.wrappers import KerasRegressor
from scipy.signal import medfilt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score


class EEGRegressor:
    def __init__(self, x_train, y_train, x_test, y_test, c_test, seglen):
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        self.c_test=c_test

        self.seglen = seglen  # Segment length (SEGLEN)

        # Preprocessed data
        self.x_train_resampled = None
        self.y_train_resampled = None

        # Final model
        self.model = None

    def preprocess_data(self):

        self.x_train_resampled = self.x_train.reshape(-1, self.seglen, 1)
        self.x_test_resampled = self.x_test.reshape(-1, self.seglen, 1)

    def create_model(self, units=64, dropout=0.5):

        model = Sequential([
            LSTM(units, return_sequences=True, input_shape=(self.seglen, 1)),
            Dense(64, activation='relu'),
            Dropout(dropout),
            Bidirectional(LSTM(units, return_sequences=True)),
            GlobalAveragePooling1D(),
            #Dropout(dropout),
            Dense(128, activation='relu'),
            Dropout(dropout),
            Dense(64, activation='relu'),
            Dense(1)
        ])


        model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
        return model

    def train_model(self, epochs=15, batch_size=256):

        self.model = self.create_model()

        # Train the model with checkpointing
        self.model.fit(
            self.x_train_resampled, self.y_train,
            validation_data=(self.x_test_resampled, self.y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[ModelCheckpoint('model.keras', save_best_only=True),
                       EarlyStopping(patience=5, restore_best_weights=True)]
        )

    def hyperparameter_tuning(self):

        model = KerasRegressor(model=lambda units, dropout: self.create_model(units=units, dropout=dropout), verbose=2)

        print("KerasRegressor initialized.")

        param_grid = {
            'model__units': [64],
            'model__dropout': [0.3, 0.5, 0.7],
            'batch_size': [150, 256],
            'epochs': [10, 15]
        }

        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=2, error_score='raise')
        print("GridSearchCV initialized.")
        grid_result = grid.fit(self.x_train_resampled, self.y_train)


        if hasattr(grid_result, 'best_score_') and hasattr(grid_result, 'best_params_'):
            print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")
            return grid_result.best_params_
        else:
            print("Grid search did not return best parameters.")
            return None


    def evaluate_model(self):

        pred_test = self.model.predict(self.x_test_resampled).flatten()

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
        plt.show()


        return test_mae, r2

    def plot_with_predictions(self):

        pred_test = self.model.predict(self.x_test_resampled).flatten()

        for caseid in np.random.choice(np.unique(self.c_test), size=3, replace=False):
            case_mask = (self.c_test == caseid)
            case_len = np.sum(case_mask)
            if case_len == 0:
                continue
            print("y_test shape", self.y_test.shape)
            print("c_test shape", self.c_test.shape)
            print("case_mask", case_mask)
            print("case_len", case_len)
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
