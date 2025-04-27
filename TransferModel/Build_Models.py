from keras import layers, models

#✅ This gives you the feature extractor for both pretraining and fine-tuning.
def create_deepsleepnet_backbone(input_shape):
    """
    Build the core DeepSleepNet feature extractor: CNN + BiLSTM (no final output layer yet).
    """
    inputs = layers.Input(shape=input_shape)  # input_shape = (sequence_length, 1) for single IMF

    # -------- CNN Feature Extraction Block --------
    # First convolution branch (small filters)
    x = layers.Conv1D(64, kernel_size=5, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(128, kernel_size=5, padding='same', activation='relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    # -------- BiLSTM Sequence Modeling Block --------
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)

    model = models.Model(inputs, x, name='deepsleepnet_backbone')

    return model

#Input shape here would be (sequence_length, 1) because you're feeding single IMFs one at a time
def build_pretraining_model(input_shape, num_classes=3):
    """
    Build the pretraining model to classify IMF1, IMF2, IMF3.
    """
    # Backbone feature extractor
    backbone = create_deepsleepnet_backbone(input_shape)

    inputs = layers.Input(shape=input_shape)
    features = backbone(inputs)

    # -------- Pretraining Head --------
    x = layers.Dense(64, activation='relu')(features)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs, name='pretraining_imf_classifier')

    return model

#Input shape is (sequence_length, 3) — your 3 IMFs stacked along the channel axis.
def build_finetuning_model(sequence_length, backbone_weights_path=None):
    """
    Build the final model that uses three backbones (one per IMF),
    concatenates their features, passes through BiLSTM + Dense to predict BIS/DOA.
    """

    # Input: (batch_size, sequence_length, 3)
    inputs = layers.Input(shape=(sequence_length, 3))  # 3 channels for IMF1, IMF2, IMF3

    # Split into individual IMFs
    imf1 = layers.Lambda(lambda x: x[:, :, 0:1])(inputs)
    imf2 = layers.Lambda(lambda x: x[:, :, 1:2])(inputs)
    imf3 = layers.Lambda(lambda x: x[:, :, 2:3])(inputs)

    # Create three backbones (can share weights if needed)
    backbone1 = create_deepsleepnet_backbone((sequence_length, 1))
    backbone2 = create_deepsleepnet_backbone((sequence_length, 1))
    backbone3 = create_deepsleepnet_backbone((sequence_length, 1))

    # Load pretrained weights if provided
    if backbone_weights_path is not None:
        backbone1.load_weights(backbone_weights_path)
        backbone2.load_weights(backbone_weights_path)
        backbone3.load_weights(backbone_weights_path)

    # Extract features
    features1 = backbone1(imf1)
    features2 = backbone2(imf2)
    features3 = backbone3(imf3)

    # Concatenate CNN outputs
    combined_features = layers.Concatenate()([features1, features2, features3])

    # Pass through BiLSTM
    x = layers.RepeatVector(1)(combined_features)  # shape hack for LSTM input
    x = layers.Bidirectional(layers.LSTM(128))(x)

    # Dense Regression Head
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1)(x)  # Single value output (BIS or DOA)

    model = models.Model(inputs, outputs, name='finetuning_doa_predictor')

    return model
