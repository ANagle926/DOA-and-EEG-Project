"""

Best: 0.34192658765093986 using {'batch_size': 150, 'epochs': 10, 'model__dropout': 0.5, 'model__units': 64}
Best Hyperparameters: {'batch_size': 150, 'epochs': 10, 'model__dropout': 0.5, 'model__units': 64}
Creating model with units=64, dropout=0.5
Creating model with units=64, dropout=0.3
Epoch 1/10
1144/1144 ━━━━━━━━━━━━━━━━━━━━ 237s 206ms/step - loss: 10.9095 - mean_absolute_error: 10.9095 - val_loss: 5.7984 - val_mean_absolute_error: 5.7984
Epoch 2/10
1144/1144 ━━━━━━━━━━━━━━━━━━━━ 235s 206ms/step - loss: 6.0700 - mean_absolute_error: 6.0700 - val_loss: 5.6159 - val_mean_absolute_error: 5.6159
Epoch 3/10
1144/1144 ━━━━━━━━━━━━━━━━━━━━ 239s 209ms/step - loss: 5.6472 - mean_absolute_error: 5.6472 - val_loss: 5.5232 - val_mean_absolute_error: 5.5232
Epoch 4/10
1144/1144 ━━━━━━━━━━━━━━━━━━━━ 238s 208ms/step - loss: 5.2350 - mean_absolute_error: 5.2350 - val_loss: 6.8548 - val_mean_absolute_error: 6.8548
Epoch 5/10
1144/1144 ━━━━━━━━━━━━━━━━━━━━ 237s 207ms/step - loss: 4.8530 - mean_absolute_error: 4.8530 - val_loss: 7.4021 - val_mean_absolute_error: 7.4021
Epoch 6/10
1144/1144 ━━━━━━━━━━━━━━━━━━━━ 232s 202ms/step - loss: 4.6576 - mean_absolute_error: 4.6576 - val_loss: 7.3061 - val_mean_absolute_error: 7.3061
2135/2135 ━━━━━━━━━━━━━━━━━━━━ 118s 55ms/step
Test MAE: 5.523221020202548
R squared: 0.3876791648173694
/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/Regression.py:116: UserWarning: kernel_size exceeds volume extent: the volume will be zero-padded.
  pred_test[case_mask] = scipy.signal.medfilt(pred_test[case_mask], kernel_size=15)
free(): invalid next size (fast)

Process finished with exit code 134



Best: 0.4482214198733658 using {'batch_size': 80, 'epochs': 18, 'model__dropout': 0.5, 'model__units': 64}
Best Hyperparameters: {'batch_size': 80, 'epochs': 18, 'model__dropout': 0.5, 'model__units': 64}
Creating model with units=64, dropout=0.5
Creating model with units=64, dropout=0.3
Epoch 1/18
2145/2145 ━━━━━━━━━━━━━━━━━━━━ 448s 208ms/step - loss: 9.2403 - mean_absolute_error: 9.2403 - val_loss: 5.7332 - val_mean_absolute_error: 5.7332
Epoch 2/18
2145/2145 ━━━━━━━━━━━━━━━━━━━━ 439s 205ms/step - loss: 5.8862 - mean_absolute_error: 5.8862 - val_loss: 5.3010 - val_mean_absolute_error: 5.3010
Epoch 3/18
2145/2145 ━━━━━━━━━━━━━━━━━━━━ 439s 205ms/step - loss: 5.3832 - mean_absolute_error: 5.3832 - val_loss: 5.5332 - val_mean_absolute_error: 5.5332
Epoch 4/18
2145/2145 ━━━━━━━━━━━━━━━━━━━━ 450s 210ms/step - loss: 4.8900 - mean_absolute_error: 4.8900 - val_loss: 5.1277 - val_mean_absolute_error: 5.1277
Epoch 5/18
2145/2145 ━━━━━━━━━━━━━━━━━━━━ 445s 207ms/step - loss: 4.6051 - mean_absolute_error: 4.6051 - val_loss: 5.3131 - val_mean_absolute_error: 5.3131
Epoch 6/18
2145/2145 ━━━━━━━━━━━━━━━━━━━━ 450s 210ms/step - loss: 4.3603 - mean_absolute_error: 4.3603 - val_loss: 5.3486 - val_mean_absolute_error: 5.3486
Epoch 7/18
2145/2145 ━━━━━━━━━━━━━━━━━━━━ 441s 206ms/step - loss: 4.2225 - mean_absolute_error: 4.2225 - val_loss: 5.1985 - val_mean_absolute_error: 5.1985
2135/2135 ━━━━━━━━━━━━━━━━━━━━ 144s 68ms/step
Test MAE: 5.127663225171577
R squared: 0.514452172448744




Best: -0.08926351635870977 using {'batch_size': 150, 'epochs': 10, 'model__dropout': 0.3, 'model__units': 64}
best units: 64
best dropout: 0.3
best epochs: 10
best batch size: 150
Epoch 1/10
2297/2297 ━━━━━━━━━━━━━━━━━━━━ 475s 206ms/step - loss: 9.1973 - mean_absolute_error: 9.1973 - val_loss: 19.7838 - val_mean_absolute_error: 19.7838
Epoch 2/10
2297/2297 ━━━━━━━━━━━━━━━━━━━━ 471s 205ms/step - loss: 5.4052 - mean_absolute_error: 5.4052 - val_loss: 20.6557 - val_mean_absolute_error: 20.6557
Epoch 3/10
2297/2297 ━━━━━━━━━━━━━━━━━━━━ 472s 205ms/step - loss: 5.2563 - mean_absolute_error: 5.2563 - val_loss: 21.9801 - val_mean_absolute_error: 21.9801
Epoch 4/10
2297/2297 ━━━━━━━━━━━━━━━━━━━━ 467s 204ms/step - loss: 4.7141 - mean_absolute_error: 4.7141 - val_loss: 21.7363 - val_mean_absolute_error: 21.7363
Epoch 5/10
2297/2297 ━━━━━━━━━━━━━━━━━━━━ 472s 206ms/step - loss: 4.5447 - mean_absolute_error: 4.5447 - val_loss: 21.2119 - val_mean_absolute_error: 21.2119
Epoch 6/10
2297/2297 ━━━━━━━━━━━━━━━━━━━━ 471s 205ms/step - loss: 4.4676 - mean_absolute_error: 4.4676 - val_loss: 20.3749 - val_mean_absolute_error: 20.3749
3748/3748 ━━━━━━━━━━━━━━━━━━━━ 241s 64ms/step
Test MAE: 19.783828685837513
R squared: -3.1213166796560987






Best: -0.3956078886985779 using {'batch_size': 150, 'epochs': 13, 'model__dropout': 0.3, 'model__units': 64}
--- space for more training => not overfitting
---data=100


Best: 0.5409204959869385 using {'batch_size': 150, 'epochs': 10, 'model__dropout': 0.5, 'model__reg_strength': 0.001, 'model__units': 64}
I think this was the run:

Epoch 1/10
1829/1829 ━━━━━━━━━━━━━━━━━━━━ 399s 216ms/step - loss: 10.1892 - mae: 9.9733 - val_loss: 6.5271 - val_mae: 6.4067

Epoch 2/10
1829/1829 ━━━━━━━━━━━━━━━━━━━━ 398s 217ms/step - loss: 6.4049 - mae: 6.2949 - val_loss: 5.2502 - val_mae: 5.1659

Epoch 3/10
1829/1829 ━━━━━━━━━━━━━━━━━━━━ 387s 212ms/step - loss: 5.4642 - mae: 5.3889 - val_loss: 4.8998 - val_mae: 4.8450

Epoch 4/10
1829/1829 ━━━━━━━━━━━━━━━━━━━━ 385s 211ms/step - loss: 5.0392 - mae: 4.9887 - val_loss: 4.9975 - val_mae: 4.9548

Epoch 5/10
1829/1829 ━━━━━━━━━━━━━━━━━━━━ 391s 214ms/step - loss: 4.8266 - mae: 4.7860 - val_loss: 5.9254 - val_mae: 5.8867

Epoch 6/10
1829/1829 ━━━━━━━━━━━━━━━━━━━━ 393s 215ms/step - loss: 4.8823 - mae: 4.8425 - val_loss: 5.5978 - val_mae: 5.5471

Epoch 7/10
1829/1829 ━━━━━━━━━━━━━━━━━━━━ 386s 211ms/step - loss: 4.8322 - mae: 4.7816 - val_loss: 5.6051 - val_mae: 5.5589

Epoch 8/10
1829/1829 ━━━━━━━━━━━━━━━━━━━━ 393s 215ms/step - loss: 4.7063 - mae: 4.6601 - val_loss: 5.2026 - val_mae: 5.1550

Epoch 9/10
1829/1829 ━━━━━━━━━━━━━━━━━━━━ 393s 215ms/step - loss: 4.6512 - mae: 4.6020 - val_loss: 5.1896 - val_mae: 5.1435

Epoch 10/10
1829/1829 ━━━━━━━━━━━━━━━━━━━━ 380s 208ms/step - loss: 4.5707 - mae: 4.5237 - val_loss: 5.2637 - val_mae: 5.2104


loss decreases (hiccup in the middle)
mae decreases (hiccup in the middle)
val_loss keeps fluctuating
val_mae keeps fluctuating

"""