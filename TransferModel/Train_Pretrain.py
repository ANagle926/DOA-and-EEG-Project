
from keras import optimizers, callbacks
from Build_Models import build_pretraining_model
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


def train_pretrain(x_pretrain, y_pretrain, seglen=128):
    # Build pretraining model
    sequence_length = seglen
    pretrain_model = build_pretraining_model(input_shape=(sequence_length, 1))

    # Compile it
    pretrain_model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train it
    pretrain_model.fit(
        x_pretrain,     # your IMF signals (shape: [samples, sequence_length, 1])
        y_pretrain,     # labels (0,1,2 for IMF1/IMF2/IMF3)
        validation_split=0.2,
        epochs=30,
        batch_size=64,
        callbacks=[
            callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        ]
    )

    # Save only the backbone (CNN+BiLSTM weights)
    pretrain_model.get_layer('deepsleepnet_backbone').save_weights('backbone_weights.h5')
    visualize_cnn(pretrain_model)

def test_pretrain(pretrain_model, x_test, y_test):

    y_pred_probs = pretrain_model.predict(x_test)  # shape: (num_samples, 3)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)  # Get predicted class index

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["IMF1", "IMF2", "IMF3"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix: IMF Classification")
    plt.show()

    # Classification Report
    print(classification_report(y_test, y_pred_classes, target_names=["IMF1", "IMF2", "IMF3"]))

def visualize_cnn(pretrain_model):
    # ✅ Interpretation: Sharp, smooth, oscillating filters = good.

    # Find first Conv1D layer dynamically
    backbone = pretrain_model.get_layer('deepsleepnet_backbone')
    conv_layer = None
    for layer in backbone.layers:
        if isinstance(layer, backbone.layers.Conv1D):
            conv_layer = layer
            break

    if conv_layer is None:
        print("No Conv1D layer found!")
        return

    filters, biases = conv_layer.get_weights()
    print(f"Filter shape: {filters.shape}")  # (kernel_size, input_channels, num_filters)

    # Plot the filters
    n_filters = filters.shape[-1]
    plt.figure(figsize=(20, 5))
    for i in range(n_filters):
        plt.subplot(4, n_filters // 4, i + 1)
        plt.plot(filters[:, 0, i])  # Plot filter weights for 1 channel
        plt.title(f'Filter {i + 1}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
