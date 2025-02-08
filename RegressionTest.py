from Regression import EEGRegressor
from VitalDBDataset import VitalDBDataset

# Load and process dataset
dataset = VitalDBDataset(max_cases=500, srate=128)
x_train, y_train = dataset.x_train, dataset.y_train
x_test, y_test = dataset.x_test, dataset.y_test
seglen = dataset.SEGLEN


# Initialize EEGRegressor object
eeg_regressor = EEGRegressor(x_train, y_train, x_test, y_test, seglen)

# Preprocess the data
eeg_regressor.preprocess_data()

# Perform hyperparameter tuning
best_params = eeg_regressor.hyperparameter_tuning()
print("Best Hyperparameters:", best_params)


# Extract best hyperparameter values safely
best_units = best_params.get('model__units', 64)  # Default to 64 if not found
best_dropout = best_params.get('model__dropout', 0.3)
best_epochs = best_params.get('epochs', 15)
best_batch_size = best_params.get('batch_size', 126)

# Train the model with the best hyperparameters
eeg_regressor.model = eeg_regressor.create_model(units=best_units, dropout=best_dropout)
eeg_regressor.train_model(epochs=best_epochs, batch_size=best_batch_size)

# Evaluate the model performance
test_mae, r2 = eeg_regressor.evaluate_model()
print(f"Final Model - Test MAE: {test_mae}, R2 Score: {r2}")
