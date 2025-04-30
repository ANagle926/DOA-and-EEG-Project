
from keras import optimizers, callbacks
from keras.src.layers import Conv1D
from keras.src.saving import load_model
from tensorflow.python.layers import layers

from Build_Models import build_pretraining_model, create_deepsleepnet_backbone
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

def prepare_pretraining_data(x_data):
    """
    Takes (samples, 512, 3) data and returns:
    - x_pretrain: (samples*3, 512, 1)
    - y_pretrain: (samples*3,)
    """
    # Separate IMFs
    imf1 = x_data[:, :, 0]  # (samples, 512)
    imf2 = x_data[:, :, 1]
    imf3 = x_data[:, :, 2]

    # Stack all IMFs vertically
    x_pretrain = np.concatenate([
        imf1[..., np.newaxis],  # (samples, 512, 1)
        imf2[..., np.newaxis],
        imf3[..., np.newaxis]
    ], axis=0)

    # Create corresponding labels
    y_pretrain = np.concatenate([
        np.zeros(imf1.shape[0]),  # Label 0 for IMF1
        np.ones(imf2.shape[0]),   # Label 1 for IMF2
        np.full(imf3.shape[0], 2) # Label 2 for IMF3
    ], axis=0)

    x_pretrain, y_pretrain = shuffle_data(x_pretrain, y_pretrain)

    return x_pretrain, y_pretrain

def shuffle_data(x, y):
    """
    Shuffles x and y in unison.
    """
    assert len(x) == len(y), "Inputs and labels must have same number of samples"
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    return x[indices], y[indices]

def train_pretrain(x_pretrain, y_pretrain, seglen):
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
        validation_split=0.3,
        epochs=5,
        batch_size=64,
        callbacks=[
            callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        ]
    )

    # Save only the backbone (CNN+BiLSTM weights)
    pretrain_model.save('final_pretrained_model.h5')
    pretrain_model.get_layer('deepsleepnet_backbone').save_weights('backbone_weights.weights.h5')
    #pretrain_model= load_model('final_pretrained_model.h5')
    return pretrain_model

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

    # Additional Metrics
    acc = accuracy_score(y_test, y_pred_classes)
    macro_f1 = f1_score(y_test, y_pred_classes, average='macro')  # 'macro' = average equally across classes

    print(f"Overall Accuracy: {acc:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")

def visualize_cnn():
    # Build and load weights into the backbone model
    backbone_model = create_deepsleepnet_backbone(input_shape=(512, 1))
    backbone_model.load_weights('backbone_weights.weights.h5')

    print("\nInspecting backbone_model layers:")
    for idx, layer in enumerate(backbone_model.layers):
        print(f"Layer {idx}: {layer.name} ({layer.__class__.__name__})")


    # Find first Conv1D layer dynamically
    conv_layers = []
    for layer in backbone_model.layers:
        if isinstance(layer, Conv1D):
            conv_layers.append(layer)

    if not conv_layers:
        print("No Conv1D layer found!")
        return

    print(f"Found {len(conv_layers)} Conv1D layers.")

    # Plot filters for each Conv1D layer separately
    for idx, conv_layer in enumerate(conv_layers):
        filters, biases = conv_layer.get_weights()
        print(f"\nConv1D Layer {idx+1}: Filter shape = {filters.shape}")  # (kernel_size, input_channels, num_filters)

        n_filters = filters.shape[-1]

        plt.figure(figsize=(20, 5))
        for i in range(n_filters):
            plt.subplot(4, n_filters // 4, i + 1)
            plt.plot(filters[:, 0, i])  # Plot filter weights for 1 channel
            plt.title(f'Filter {i + 1}')
            plt.axis('off')
        plt.suptitle(f"Conv1D Layer {idx+1} Filters", fontsize=16)
        plt.tight_layout()
        plt.show()

