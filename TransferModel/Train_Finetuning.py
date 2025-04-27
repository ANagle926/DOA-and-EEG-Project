from keras import optimizers, callbacks
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

from Build_Models import build_finetuning_model
import numpy as np
import matplotlib.pyplot as plt


def train_finetuning(x_finetune, y_finetune, seglen=128):

    sequence_length =seglen

    # Build fine-tuning model
    finetune_model = build_finetuning_model(sequence_length=sequence_length, backbone_weights_path='backbone_weights.h5')

    # Compile it
    finetune_model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss='mean_squared_error',
        metrics=['mae']
    )

    #  Freeze the CNNs initially
    for layer in finetune_model.layers:
        if 'deepsleepnet_backbone' in layer.name:
            layer.trainable = False

    # Train the Dense/LSTM heads first (warm-up training)
    finetune_model.fit(
        x_finetune,   # your (IMF1, IMF2, IMF3) stacked data
        y_finetune,   # BIS or DOA targets
        validation_split=0.2,
        epochs=10,
        batch_size=64,
        callbacks=[
            callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        ]
    )

    # Unfreeze everything
    for layer in finetune_model.layers:
        layer.trainable = True

    # Continue fine-tuning whole model
    finetune_model.fit(
        x_finetune,
        y_finetune,
        validation_split=0.2,
        epochs=50,
        batch_size=64,
        callbacks=[
            callbacks.ModelCheckpoint('best_finetuned_model.h5', save_best_only=True, monitor='val_mae'),
            callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6)
        ]

    )

    finetune_model.save('final_finetuned_model.h5')

def test_finetuning(finetune_model, x_test, y_test):
    # Predict
    pred_test = finetune_model.predict(x_test).flatten()  # make sure it's 1D array


    test_mae = mean_absolute_error(y_test, pred_test)
    corr = np.corrcoef(y_test, pred_test)[0, 1]
    r2 = r2_score(y_test, pred_test)
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Correlation coefficient: {corr:.4f}")
    print(f"R squared: {r2:.4f}")

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



