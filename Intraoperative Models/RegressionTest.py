from Regression import EEGRegressor
from joblib import dump, load


from VitalDBDataset import VitalDBDataset

dataset = VitalDBDataset(max_cases=200, srate=128)
dump(dataset, "/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/Data Files/dataset.joblib")

"""
#dataset = load("../Data Files/Pre_processed_Data.joblib")
x_train, y_train = dataset.x_train, dataset.y_train
x_test, y_test = dataset.x_test, dataset.y_test
c_train, c_test = dataset.c_train, dataset.c_test
seglen = dataset.SEGLEN


# Initialize EEGRegressor object
eeg_regressor = EEGRegressor(x_train, y_train, x_test, y_test, c_test, seglen)

# Perform hyperparameter tuning
best_params= eeg_regressor.hyperparameter_tuning()

# Extract best hyperparameter values
best_units = best_params.get('model__units', 64)
best_dropout = best_params.get('model__dropout', 0.3)
best_reg_strength = best_params.get('model__reg_strength', 0.001)
best_learning_rate = best_params.get('model__learning_rate', 0.001)
best_epochs = best_params.get('epochs', 15)
best_batch_size = best_params.get('batch_size', 150)

print("best units:", best_units)
print("best dropout:", best_dropout)
print("best reg strength:", best_reg_strength)
print("best learning rate:", best_learning_rate)
print("best epochs:", best_epochs)
print("best batch size:", best_batch_size)


# Train the model with the best hyperparameters
eeg_regressor.model = eeg_regressor.create_model(units=best_units, dropout=best_dropout, reg_strength=best_reg_strength, learning_rate=best_learning_rate)
eeg_regressor.train_model(epochs=best_epochs, batch_size=best_batch_size)
eeg_regressor.model.save("eeg_regressor.keras")

# Evaluate the model performance
test_mae, r2 = eeg_regressor.evaluate_model()
print(f"Final Model - Test MAE: {test_mae}, R2 Score: {r2}")"""
