from Regression import EEGRegressor

from joblib import dump, load

from VitalDBDataset import VitalDBDataset
#dataset = VitalDBDataset(max_cases=20, srate=128)
#dump(dataset, "/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/Data Files/dataset_twenty_cases.joblib")

dataset=load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/Data Files/dataset_twenty_cases.joblib")

x_train, y_train = dataset.x_train, dataset.y_train
x_test, y_test = dataset.x_test, dataset.y_test
c_train, c_test = dataset.c_train, dataset.c_test
seglen = dataset.SEGLEN


# Initialize EEGRegressor object
eeg_regressor = EEGRegressor(x_train, y_train, x_test, y_test, c_test, seglen)

# Perform hyperparameter tuning
best_params= eeg_regressor.hyperparameter_tuning()
model= eeg_regressor.model
dump(model, "/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/Data Files/Model Versions/eeg_regressor.joblib")

# Extract best hyperparameter values
best_units = best_params.get('model__units')
best_dropout = best_params.get('model__dropout')
best_reg_strength = best_params.get('model__reg_strength')
best_learning_rate = best_params.get('model__learning_rate')
best_epochs = best_params.get('epochs')
best_batch_size = best_params.get('batch_size')

print("best units:", best_units)
print("best dropout:", best_dropout)
print("best reg strength:", best_reg_strength)
print("best learning rate:", best_learning_rate)
print("best epochs:", best_epochs)
print("best batch size:", best_batch_size)

#eeg_regressor.train_model()
eeg_regressor.evaluate_model()