import keras
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from imblearn.tensorflow.tests.test_generator import tf

from keras import Sequential
from keras.src.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.src.layers import LSTM, Dense, Dropout, Bidirectional, GlobalAveragePooling1D, LayerNormalization, \
    GlobalMaxPooling1D, MaxPooling1D, Conv1D, SpatialDropout1D
from keras.src.optimizers import Adam

from scikeras.wrappers import KerasRegressor
from scipy.signal import medfilt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from keras.src.callbacks import ReduceLROnPlateau, EarlyStopping



class ValidationLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs:
            print(f"Epoch {epoch+1}: val_mae = {logs.get('val_mean_absolute_error', 'Not Available')}")


class EEGRegressor:
    def __init__(self, x_train, y_train, x_test, y_test, c_test, seglen):
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        self.c_test=c_test
        self.seglen = seglen
        self.model = None

    def create_model(self, units=64, dropout=0, reg_strength=0.001, learning_rate=0.0005):
        model = Sequential([
            # Optional: local pattern extraction
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(self.seglen, 3)),
            SpatialDropout1D(dropout),
            Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(reg_strength)),
            #add layer normalization here?
            MaxPooling1D(pool_size=2),
            Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(reg_strength)),
            Bidirectional(LSTM(units, return_sequences=True, kernel_regularizer=keras.regularizers.l2(reg_strength))),
            LayerNormalization(),
            LSTM(units, return_sequences=True, kernel_regularizer=keras.regularizers.l2(reg_strength)),
            GlobalAveragePooling1D(),
            Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(reg_strength)),
            LayerNormalization(),
            Dropout(dropout),
            Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(reg_strength)),
            Dense(1)
        ])

        optimizer = Adam(learning_rate=learning_rate)  # Use learning_rate from GridSearch
        model.compile(loss='mae', optimizer=optimizer, metrics=['mae'])
        return model

    #Dont use train_model for hyperparameter tuning
    def train_model(self, epochs=90, batch_size=64):
        self.model = self.create_model()
        reduce_lr = ReduceLROnPlateau(
            monitor='val_mae',
            factor=0.5,            # Reduce LR by a factor of 0.5
            patience=4,            # Wait 4 epochs with no improvement
            min_lr=1e-6,           # Don't go below this learning rate
            verbose=1              # Print when LR is reduced
        )
        early_stop = EarlyStopping(   
            monitor='val_mae',        
            patience=8,               
            restore_best_weights=True,
            verbose=1                 
        )                               
        self.model.fit(
            self.x_train, self.y_train,
            validation_data=(self.x_test, self.y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                ModelCheckpoint('../Data Files/Model Versions/model.keras', save_best_only=True),
                early_stop,
                reduce_lr
            ]
        )

    def hyperparameter_tuning(self):
        # Define your callbacks
        reduce_lr = ReduceLROnPlateau(
            monitor='val_mae',
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1
        )

        early_stop = EarlyStopping(
            monitor='val_mae',
            patience=8,
            restore_best_weights=True,
            verbose=1
        )

        # Initialize KerasRegressor
        model = KerasRegressor(
            model=lambda units, dropout, reg_strength, learning_rate: self.create_model(
                units=units, dropout=dropout, reg_strength=reg_strength, learning_rate=learning_rate
            ),
            verbose=1
        )

        print("KerasRegressor initialized.")

        param_grid = {
            'batch_size': [64],
            'epochs': [90],
            'model__dropout': [0],  # always zero, don't change
            'model__learning_rate': [0.001],
            'model__reg_strength': [0.001],
            'model__units': [64],
        }

        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring='neg_mean_absolute_error',
            n_jobs=1,
            cv=2,
            return_train_score=True,
            error_score='raise',
            verbose=2
        )

        print("GridSearchCV initialized.")

        grid_result = grid.fit(
            self.x_train,
            self.y_train,
            **{
                "validation_data": (self.x_test, self.y_test),
                "callbacks": [ValidationLogger(), reduce_lr, early_stop]
            }
        )

        if hasattr(grid_result, 'best_score_') and hasattr(grid_result, 'best_params_'):
            print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")
            self.model = grid_result.best_estimator_.model_
            return grid_result.best_params_
        else:
            print("Grid search did not return best parameters.")
            return None


    def moving_avg(self, signal, window=15):
        return np.convolve(signal, np.ones(window)/window, mode='same')

    def evaluate_model(self):

        # Predict and evaluate test statistics
        pred_test = self.model.predict(self.x_test).flatten()

        test_mae = mean_absolute_error(self.y_test, pred_test)
        corr = np.corrcoef(self.y_test, pred_test)[0, 1]
        r2 = r2_score(self.y_test, pred_test)
        print(f"Test MAE: {test_mae:.4f}")
        print(f"Correlation coefficient: {corr:.4f}")
        print(f"R squared: {r2:.4f}")


        # 1. Scatter plot: Actual vs Predicted
        plt.figure(figsize=(6, 6))
        plt.scatter(self.y_test, pred_test, s=1, alpha=0.5, color='violet')
        plt.xlabel('Actual BIS')
        plt.ylabel('Predicted BIS')
        plt.title(f'Scatter Plot (Correlation: {corr:.4f})')
        plt.plot([0, max(self.y_test)], [0, max(self.y_test)], 'r--')
        plt.grid(True)
        plt.show()

        # 2. Histogram of prediction errors
        errors = self.y_test - pred_test
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
        sc = plt.scatter(self.y_test, pred_test, c=abs_errors, s=2, cmap='viridis', alpha=0.6)
        plt.xlabel('Actual BIS')
        plt.ylabel('Predicted BIS')
        plt.title('Scatter Plot Colored by Absolute Error')
        plt.colorbar(sc, label='Absolute Error')
        plt.plot([0, max(self.y_test)], [0, max(self.y_test)], 'r--')
        plt.grid(True)
        plt.show()

        # 4. Time-series plots with moving average overlay (3 random cases)
        for caseid in np.random.choice(np.unique(self.c_test), size=1, replace=False):
            case_mask = (self.c_test == caseid)
            case_len = np.sum(case_mask)
            if case_len == 0:
                continue

            actual = self.y_test[case_mask]
            predicted = pred_test[case_mask]
            t = np.arange(case_len)
            mae = np.mean(np.abs(actual - predicted))

            plt.figure(figsize=(12, 4))
            plt.plot(t, actual, label='Actual BIS', color='black')
            plt.plot(t, predicted, label='Predicted BIS', color='royalblue', alpha=0.7)
            plt.plot(t, self.moving_avg(predicted), label='Predicted (Moving Avg)', linestyle='--', color='orange')
            plt.xlabel('Time')
            plt.ylabel('BIS')
            plt.title(f'Case {caseid} | MAE: {mae:.4f}')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
