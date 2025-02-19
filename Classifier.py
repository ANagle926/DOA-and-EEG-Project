import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from keras.src.callbacks import ModelCheckpoint
from keras.src.layers import LSTM, Dense, Dropout, Bidirectional, GlobalAveragePooling1D
from keras.src.utils import to_categorical
from keras import Sequential

from imblearn.over_sampling import SMOTE
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, f1_score


class EEGClassifier:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

        # Preprocessed data
        self.y_train_cat = None
        self.y_test_cat = None
        self.x_train_resampled = None
        self.y_train_resampled = None

        # Final Model
        self.model = None

    def preprocess_data(self):
        """Convert labels to categories, apply SMOTE, and one-hot encode."""
        self.y_train_cat = self.convert_to_categories(self.y_train)
        self.y_test_cat = self.convert_to_categories(self.y_test)

        # Flatten x_train for SMOTE
        x_train_flat = self.x_train.reshape(self.x_train.shape[0], -1)

        # Apply SMOTE for class balancing
        smote = SMOTE(random_state=42)
        self.x_train_resampled, self.y_train_resampled = smote.fit_resample(x_train_flat, self.y_train_cat)
        self.x_train_resampled = self.x_train_resampled.reshape(-1, self.x_train.shape[1], 1)

        # Convert labels to one-hot encoding
        self.y_train_cat = to_categorical(self.y_train_resampled, num_classes=3)
        self.y_test_cat = to_categorical(self.y_test_cat, num_classes=3)

        print("Original class distribution:", Counter(self.y_train_cat))
        print("Resampled class distribution:", Counter(self.y_train_resampled))

    @staticmethod
    def convert_to_categories(y):
        """Convert numeric values into discrete categories."""
        categories = []
        for value in y:
            if value < 40:
                categories.append(0)  # AD
            elif 40 <= value <= 60:
                categories.append(1)  # AO
            else:
                categories.append(2)  # AL
        return np.array(categories)

    @staticmethod
    def create_model(units=64, dropout=0.3):
        """Create an LSTM-based deep learning model."""
        model = Sequential([
            LSTM(units, return_sequences=True),
            Dense(64, activation='relu'),
            Dropout(dropout),
            Bidirectional(LSTM(units, return_sequences=True)),
            GlobalAveragePooling1D(),
            Dropout(dropout),
            Dense(100, activation='relu'),
            Dropout(dropout),
            Dense(64, activation='relu'),
            Dense(3, activation='softmax')
        ])
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train_model(self, epochs=15, batch_size=126):
        """Train the model using the processed dataset."""
        self.model = self.create_model()

        # Compile the model
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Train the model with checkpointing
        self.model.fit(
            self.x_train_resampled, self.y_train_cat,
            validation_data=(self.x_test, self.y_test_cat),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[ModelCheckpoint('model.keras', save_best_only=True)]
        )

    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning using GridSearchCV."""
        model = KerasClassifier(model=self.create_model, verbose=1)

        param_grid = {
            'model__units': [40, 64],
            'model__dropout': [0.4, 0.5, 0.6],
            'batch_size': [100, 150, 200],
            'epochs': [10, 15]
        }

        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=2)
        grid_result = grid.fit(self.x_train_resampled, self.y_train_cat)

        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        return grid_result.best_params_

    def evaluate_model(self):
        """Evaluate the model performance and generate metrics."""
        pred_class_test_prob = self.model.predict(self.x_test)
        pred_class_test = np.argmax(pred_class_test_prob, axis=1)
        y_test_labels = np.argmax(self.y_test_cat, axis=1)

        # Accuracy and F1 Score
        accuracy = accuracy_score(y_test_labels, pred_class_test)
        f1 = f1_score(y_test_labels, pred_class_test, average='macro')

        # Confusion Matrix
        cm = confusion_matrix(y_test_labels, pred_class_test)
        print(f'Classification Accuracy: {accuracy:.4f}')
        print("\nConfusion Matrix:\n", cm)
        print("\nClassification Report:\n", classification_report(y_test_labels, pred_class_test, target_names=['AD', 'AO', 'AL']))
        print(f"\nF1 Score (macro): {f1:.4f}")

        # ROC and AUC
        n_classes = 3
        fpr, tpr, roc_auc = dict(), dict(), dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(self.y_test_cat[:, i], pred_class_test_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        for i in range(n_classes):
            print(f"AUC for class {i}: {roc_auc[i]:.4f}")

        # Plot Confusion Matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()

        return accuracy, cm, f1, roc_auc
