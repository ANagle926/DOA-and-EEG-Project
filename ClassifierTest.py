from Classifier import EEGClassifier
from VitalDBDataset import VitalDBDataset

# Load and process dataset
dataset = VitalDBDataset(max_cases=500, srate=128)
x_train, y_train = dataset.x_train, dataset.y_train
x_test, y_test = dataset.x_test, dataset.y_test


# Initialize EEGClassifier object
eeg_classifier = EEGClassifier(x_train, y_train, x_test, y_test, seglen)

# Preprocess the data
eeg_classifier.preprocess_data()

# Perform hyperparameter tuning
best_params = eeg_classifier.hyperparameter_tuning()
print("Best Hyperparameters:", best_params)



# Extract correct hyperparameter keys
best_units = best_params.get('model__units', 64)  # Default to 64 if not found
best_dropout = best_params.get('model__dropout', 0.3)
best_epochs = best_params.get('epochs', 15)
best_batch_size = best_params.get('batch_size', 126)

# Train the model with the best hyperparameters
eeg_classifier.model = eeg_classifier.create_model(units=best_units, dropout=best_dropout)


eeg_classifier.train_model(epochs=best_params['epochs'], batch_size=best_params['batch_size'])

# Evaluate the model performance
accuracy, cm, f1, roc_auc = eeg_classifier.evaluate_model()
print(f"Final Model - Accuracy: {accuracy},CM: {cm}, F1: {f1}, Test roc_auc: {roc_auc},")

