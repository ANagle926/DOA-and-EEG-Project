import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

from keras import Sequential
from keras.src.callbacks import ModelCheckpoint, EarlyStopping
from keras.src.layers import LSTM, Dense, Dropout, Bidirectional, GlobalAveragePooling1D

from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score


class EEGRegressor:
    def __init__(self, x_train, y_train, x_test, y_test, seglen):
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        self.seglen = seglen  # Segment length (SEGLEN)

        # Preprocessed data
        self.x_train_resampled = None
        self.y_train_resampled = None

        # Final model
        self.model = None

    def preprocess_data(self):
        """Preprocess the data (e.g., resample, normalization, etc.)."""
        # Reshape the input data for training
        self.x_train_resampled = self.x_train.reshape(-1, self.seglen, 1)
        self.x_test_resampled = self.x_test.reshape(-1, self.seglen, 1)

    def create_model(self, units=64, dropout=0.3):
        """Create the LSTM regression model."""
        model = Sequential([
            LSTM(units, return_sequences=True, input_shape=(self.seglen, 1)),
            Dense(64, activation='relu'),
            Dropout(dropout),
            Bidirectional(LSTM(units, return_sequences=True)),
            GlobalAveragePooling1D(),
            Dropout(dropout),
            Dense(100, activation='relu'),
            Dropout(dropout),
            Dense(32, activation='relu'),
            Dense(1)  # Output layer for regression
        ])
        model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
        return model

    def train_model(self, epochs=15, batch_size=256):
        """Train the LSTM model."""
        self.model = self.create_model()

        # Train the model with checkpointing
        self.model.fit(
            self.x_train_resampled, self.y_train,
            validation_data=(self.x_test_resampled, self.y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[ModelCheckpoint('model.keras', save_best_only=True),
                       EarlyStopping(patience=3, restore_best_weights=True)]
        )

    def hyperparameter_tuning(self):
        """Hyperparameter tuning using GridSearchCV."""
        model = KerasRegressor(model=self.create_model, verbose=1)

        param_grid = {
            'model__units': [64, 100],
            'model__dropout': [0.3, 0.5],
            'batch_size': [150, 200, 256],
            'epochs': [10, 15]
        }

        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=2)
        grid_result = grid.fit(self.x_train_resampled, self.y_train)

        print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")
        return grid_result.best_params_

    def evaluate_model(self):
        """Evaluate and analyze the model's performance."""
        # Predict the test set
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

        # Optionally smooth the predictions (median filter)
        for caseid in np.unique(self.y_test):
            case_mask = (self.y_test == caseid)
            pred_test[case_mask] = scipy.signal.medfilt(pred_test[case_mask], kernel_size=15)

        # Plot actual vs predicted after smoothing
        plt.figure(figsize=(10, 6))
        plt.plot(self.y_test, label='Actual')
        plt.plot(pred_test, label='Predicted (Smoothed)', linestyle='--')
        plt.legend()
        plt.xlabel('Sample')
        plt.ylabel('Value')
        plt.title('Model Predictions vs Actual (Smoothed)')
        plt.show()

        return test_mae, r2
