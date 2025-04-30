from keras import optimizers, callbacks
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

from Build_Models import build_finetuning_model, build_early_fusion_model
import numpy as np
import matplotlib.pyplot as plt


def train_finetuning(x_finetune, y_finetune, seglen):
    sequence_length =seglen

    # Build fine-tuning model
    finetune_model = build_finetuning_model(sequence_length=sequence_length, backbone_weights_path='backbone_weights.weights.h5')
    #finetune_model = build_early_fusion_model(sequence_length=sequence_length)

    # Compile it
    finetune_model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss='mean_squared_error',
        metrics=['mae']
    )

    #  Freeze the CNNs initially
    for layer in finetune_model.layers:
        if 'deepsleepnet_backbone' in layer.name:
            print("freezing")
            layer.trainable = False

    # Train the Dense/LSTM heads first (warm-up training)
    finetune_model.fit(
        x_finetune,   # your (IMF1, IMF2, IMF3) stacked data
        y_finetune,   # BIS or DOA targets
        validation_split=0.2,
        epochs=40,
        batch_size=128,
        callbacks=[
            callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(factor=0.8, patience=6, min_lr=1e-6)
        ]
    )

    # Unfreeze everything
    print("unfreezing")
    for layer in finetune_model.layers:
        layer.trainable = True

    # Continue fine-tuning whole model
    finetune_model.fit(
        x_finetune,
        y_finetune,
        validation_split=0.2,
        epochs=60,
        batch_size=128,
        callbacks=[
            callbacks.ModelCheckpoint('best_finetuned_model.keras', save_best_only=True, monitor='val_mae'),
            callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(factor=0.8, patience=6, min_lr=1e-6)
        ]

    )

    finetune_model.save('final_finetuned_model.h5')
    return finetune_model

def test_finetuning(finetune_model, x_test, y_test):

    #pred_test = finetune_model.predict(x_test).flatten()  # make sure it's 1D array
    pred_test = mc_dropout_predict(finetune_model, x_test, n_samples=10)
    pred_test = np.squeeze(pred_test)
    apply_calibration(pred_test, y_test)

    # Histogram of prediction errors
    errors = y_test - pred_test
    plt.figure(figsize=(8, 4))
    plt.hist(errors, bins=50, color='steelblue', edgecolor='black')
    plt.xlabel('Prediction Error (Actual - Predicted)')
    plt.ylabel('Count')
    plt.title('Histogram of Prediction Errors')
    plt.grid(True)
    plt.show()

    #Scatter plot colored by absolute error
    abs_errors = np.abs(errors)
    plt.figure(figsize=(6, 6))
    sc = plt.scatter(y_test, pred_test, c=abs_errors, s=2, cmap='viridis', alpha=0.6)
    plt.xlabel('Actual BIS')
    plt.ylabel('Predicted BIS')
    plt.title('Scatter Plot Colored by Absolute Error')
    plt.colorbar(sc, label='Absolute Error')
    plt.plot([0, max(y_test)], [0, max(y_test)], 'r--')
    plt.grid(True)
    plt.show()

    # Line Plot: True vs Predicted across samples
    plt.figure(figsize=(14, 6))
    plt.plot(y_test, label='True')
    plt.plot(pred_test, label='Predicted')
    plt.xlabel('Sample Index')
    plt.ylabel('BIS/DOA Value')
    plt.title('Predicted vs True BIS/DOA Over Samples')
    plt.legend()
    plt.grid(True)
    plt.show()

def apply_calibration(y_pred, y_test, a_tol=0.2, b_tol=10):
    # Reshape because sklearn expects 2D input
    y_pred_reshaped = y_pred.reshape(-1, 1)

    # Fit linear model
    calibration_model = LinearRegression()
    calibration_model.fit(y_pred_reshaped, y_test)

    # Get a, b
    a = calibration_model.coef_[0]
    b = calibration_model.intercept_

    slope_ok = (1 - a_tol) <= a <= (1 + a_tol)
    bias_ok = abs(b) <= b_tol

    if slope_ok and bias_ok:
        print(f"Calibration OK ✅ | Slope: {a:.4f}, Bias: {b:.4f}")
    else:
        print(f"⚠️ Calibration Warning ⚠️ | Slope: {a:.4f}, Bias: {b:.4f}")
        if not slope_ok:
            print(f"  - Slope deviates too much from 1 (acceptable range: {1-a_tol:.2f} to {1+a_tol:.2f})")
        if not bias_ok:
            print(f"  - Bias is too large (acceptable bias: ±{b_tol})")

    # Apply calibration
    y_test_pred_calibrated = a * y_pred + b

    test_mae = mean_absolute_error(y_test, y_test_pred_calibrated)
    corr = np.corrcoef(y_test, y_test_pred_calibrated)[0, 1]
    r2 = r2_score(y_test, y_test_pred_calibrated)
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Correlation coefficient: {corr:.4f}")
    print(f"R squared: {r2:.4f}")

import numpy as np

def mc_dropout_predict(model, x_test, n_samples=10, batch_size=128):
    """
    Make MC Dropout predictions by averaging over n_samples stochastic forward passes.
    Processes in batches to avoid GPU OOM.
    """
    preds = []

    for sample_idx in range(n_samples):
        batch_preds = []

        # Predict in small batches
        for i in range(0, x_test.shape[0], batch_size):
            x_batch = x_test[i:i+batch_size]
            pred_batch = model(x_batch, training=True)
            batch_preds.append(pred_batch.numpy())

        batch_preds = np.concatenate(batch_preds, axis=0)  # (full test size, output_dim)
        preds.append(batch_preds)

    preds = np.stack(preds, axis=0)  # (n_samples, full test size, output_dim)
    mean_preds = np.mean(preds, axis=0)  # Average over MC samples

    return mean_preds




