import numpy as np
from keras import layers, models, regularizers

class MCDropout(layers.Dropout):
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)

def build_early_fusion_pretraining_model(sequence_length, num_classes=2):
    """
    Early fusion pretraining model using 3-channel IMF input
    for noise classification (clean vs noisy EEG).
    """
    inputs = layers.Input(shape=(sequence_length, 3))  # IMF1, IMF2, IMF3 as channels

    x = layers.GaussianNoise(0.03)(inputs)

    # CNN Feature Extractor (DeepSleepNet-inspired)
    x = layers.Conv1D(64, kernel_size=5, padding='same', activation='relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(128, kernel_size=5, padding='same', activation='relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(256, kernel_size=5, padding='same', activation='relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.LayerNormalization()(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)

    # Classification Head
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)  # Binary: clean/noisy

    model = models.Model(inputs, outputs, name="early_fusion_noise_classifier")
    return model