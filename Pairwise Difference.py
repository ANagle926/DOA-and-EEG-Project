import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.ensemble import BalancedBaggingClassifier, EasyEnsembleClassifier
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE
from joblib import dump, load
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from VitalDBDataset import VitalDBDataset


pd.set_option('display.max_columns', None)

# stop wrapping into multiple blocks
pd.set_option('display.expand_frame_repr', False)

# widen the “page” so it doesn’t wrap
#pd.set_option('display.width', 200)   # or None for auto



"""dataset= VitalDBDataset(num_cases=5000, threshold_def=0.1)

x_train= dataset.x_train
x_test= dataset.x_test
y_train= dataset.y_train
y_test= dataset.y_test
c_train= dataset.c_train
c_test= dataset.c_test

dump(x_train, "/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/x_train_preop3.joblib")
dump(x_test, "/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/x_test_preop3.joblib")
dump(y_train, "/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/y_train_preop3.joblib")
dump(y_test, "/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/y_test_preop3.joblib")
dump(c_train, "/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/c_train_preop3.joblib")
dump(c_test, "/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/c_test_preop3.joblib")"""

x_train= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/x_train_preop3.joblib")
x_test= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/x_test_preop3.joblib")
y_train= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/y_train_preop3.joblib")
y_test= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/y_test_preop3.joblib")

# 1.1. Your raw data
X, y = x_train, y_train
n_classes = len(np.unique(y))

# 1.2. Base models
base_models = {
    'bb': BalancedBaggingClassifier(estimator=DecisionTreeClassifier(max_depth=6), sampling_strategy="auto", n_estimators=10,max_samples=0.5, replacement=True, random_state=42, n_jobs=1),
    'svc': SVC(random_state=42, probability=True),
    'knn': KNeighborsClassifier(),
    'ee': EasyEnsembleClassifier(n_estimators=30,  random_state=42),
    'svc2': SVC(random_state=42, probability=True),
    'bb2': BalancedBaggingClassifier(estimator=DecisionTreeClassifier(max_depth=6), sampling_strategy="auto", n_estimators=10,max_samples=0.5, replacement=True, random_state=42, n_jobs=1),
    'bb3': BalancedBaggingClassifier(estimator=DecisionTreeClassifier(max_depth=6), sampling_strategy="auto", n_estimators=10,max_samples=0.5, replacement=True, random_state=42, n_jobs=1),
    'sgd': SGDClassifier(loss="log_loss", penalty="elasticnet",class_weight="balanced", random_state=42),
    'qda':QuadraticDiscriminantAnalysis(reg_param=0.01),
    'tabnet':TabNetClassifier(n_d=32, n_steps=5, gamma=1.5, seed=42, verbose=0),
    'tabnet2':TabNetClassifier(n_d=32, n_steps=5, gamma=1.5, seed=42, verbose=0),
}

# 1.3. A separate SMOTE (or variant) for each model
smote_map = {
    'bb': SMOTE(sampling_strategy="not majority", random_state=42),
    'svc': SMOTE(sampling_strategy="not majority", random_state=42),
    'knn': BorderlineSMOTE(kind="borderline-2", random_state=42),
    'ee':  SVMSMOTE(sampling_strategy="not majority", random_state=42),
    'svc2':SMOTE(random_state=42),
    'bb2': BorderlineSMOTE(kind="borderline-2", random_state=42),
    'bb3': SMOTEENN(sampling_strategy="not majority"),
    'sgd': SMOTE(random_state=42),
    'qda': BorderlineSMOTE(kind="borderline-2", random_state=42),
    'tabnet':SVMSMOTE(sampling_strategy="not majority", random_state=42),
    'tabnet2': SMOTEENN(sampling_strategy="not majority"),
}

# 2.1. Prepare storage
oof_preds = {
    name: np.zeros((len(y), n_classes))
    for name in base_models
}
y_true = y.copy()

# 2.2. Stratified K-fold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in base_models.items():
    sm = smote_map[name]
    for train_idx, val_idx in kf.split(X, y):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx],   y[val_idx]

        # 2.3. Resample *only* the training fold
        X_res, y_res = sm.fit_resample(X_tr, y_tr)

        # 2.4. Fit & predict
        model.fit(X_res, y_res)
        oof_preds[name][val_idx] = model.predict_proba(X_val)

# 3.1. Turn probabilities into class-labels
pred_labels = {
    name: oof_preds[name].argmax(axis=1)
    for name in base_models
}

# 3.2. Disagreement rate
def disagreement_rate(a, b):
    return np.mean(a != b)

models = list(base_models)
m = len(models)
disc = np.zeros((m, m))

for i, mi in enumerate(models):
    for j, mj in enumerate(models):
        disc[i, j] = disagreement_rate(pred_labels[mi], pred_labels[mj])

print("Disagreement rate:")
print(pd.DataFrame(disc, index=models, columns=models))


# 3.3. Binary error vectors
err = {
    name: (pred_labels[name] != y_true).astype(int)
    for name in base_models
}

# 3.4. Pearson correlation of their errors
err_corr = np.corrcoef([err[name] for name in models])
print("Error‐correlation:")
print(pd.DataFrame(err_corr, index=models, columns=models))


#If you see many pairs with low disagreement (say < 0.2)
# or high error‐corr (> 0.8)
# consider dropping one of them or adding a qualitatively different learner.

