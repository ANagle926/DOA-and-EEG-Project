from collections import Counter

import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.over_sampling import SMOTE
from joblib import load, dump
from matplotlib import pyplot as plt
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

"""x_test = load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/x_test_preop.joblib")
y_test = load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/y_test_preop.joblib")
x_train= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/x_train_preop.joblib")
y_train= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/y_train_preop.joblib")

# Define full feature names
static_cols = ["gender", "bmi", "htn", "dm", "anemia", "ph", "creatinine", "gpt", "paO2", "paCO2"]
surg_cols   = [
    "op_Biliary/Pancreas", "op_Breast", "op_Colorectal", "op_Hepatic",
    "op_Major resection", "op_Minor resection", "op_Others", "op_Stomach",
    "op_Thyroid", "op_Transplantation", "op_Vascular"
]

all_cols = static_cols + surg_cols

numeric_feats  = ["bmi", "ph", "creatinine", "gpt", "paO2", "paCO2"]
categorical_feats = ["gender","htn","dm","anemia"] + surg_cols

# Build preprocessor
preprocessor = ColumnTransformer([
    ("num", StandardScaler(),    numeric_feats),
    ("cat", OneHotEncoder(handle_unknown='ignore'),     categorical_feats),
])

# Convert x_train/x_test into DataFrames
df_train = pd.DataFrame(x_train, columns=all_cols)
df_test  = pd.DataFrame(x_test,  columns=all_cols)

X_train_proc = preprocessor.fit_transform(df_train)
X_test_proc  = preprocessor.transform(df_test)

if sparse.issparse(X_train_proc):
    X_train_proc = X_train_proc.toarray()
    X_test_proc  = X_test_proc.toarray()

# SMOTE which over-samples the training set
print(f"Original training dataset shape: {Counter(y_train)}")
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train_proc, y_train)
print(f"Resampled training dataset shape: {Counter(y_res)}")

dump(X_res, "/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/x_smote_preop.joblib")
dump(y_res, "/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/y_smote_preop.joblib")
dump(X_train_proc, "/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/x_train_proc.joblib")
dump(X_test_proc, "/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/x_test_proc.joblib")"""

X_test_proc= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/x_test_proc.joblib")
X_res= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/x_smote_preop.joblib")
y_res= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/y_smote_preop.joblib")
y_test = load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/y_test_preop.joblib")

"""ensemble = VotingClassifier(
    estimators=[
        ('bb',  BalancedBaggingClassifier(estimator=DecisionTreeClassifier(max_depth=6), sampling_strategy="auto", n_estimators=10,max_samples=0.5, replacement=True, random_state=42, n_jobs=-1)),
        ('xgb',  XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42, use_label_encoder=False, eval_metric="mlogloss")),
        ('dt',  DecisionTreeClassifier(random_state=42))
    ],
    voting='soft',
    weights=[1.7, 1, 1],  # you can increase the weight of the model that does best on class 1
    n_jobs=-1
)"""

"""stack = StackingClassifier(
    estimators=[
        ('xgb',  XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42, use_label_encoder=False, eval_metric="mlogloss")),
        ('dt',  DecisionTreeClassifier(random_state=42))
    ],
    final_estimator=BalancedBaggingClassifier(estimator=DecisionTreeClassifier(max_depth=6), sampling_strategy="auto", n_estimators=10,max_samples=0.5, replacement=True, random_state=42, n_jobs=-1),
    cv=5,
    n_jobs=-1
)"""

bb = BalancedBaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=None, class_weight='balanced'),
    n_estimators=100,
    max_samples=1.0,      # each tree sees *all* classes
    bootstrap=False,      # no duplicates, better minority retention
    random_state=42,
    n_jobs=1              # keep nested parallel disabled
)
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    random_state=42,
    eval_metric='mlogloss'
)
dt = DecisionTreeClassifier(max_depth=None,
                            class_weight='balanced',   # <-- magic flag
                            random_state=42)

stack = StackingClassifier(
    estimators=[
        ('xgb',  xgb),
        ('dt', dt),
        ('bb', bb)
    ],
    final_estimator=LogisticRegression(max_iter=10_000, class_weight="balanced"), #make this xgb?
    #make final estimator minority sensitive: BalancedRandomForestClassifie, LightGBM, CatBoost
    stack_method="predict_proba",
    cv=5,
    n_jobs=-1
)

from sklearn.model_selection import train_test_split
X_fit, _, y_fit, _ = train_test_split(
    X_res, y_res,
    train_size=5000,           # or whatever size you want (8000/3000)
    stratify=y_res,
    random_state=42
)

"""print("fitting")
ensemble.fit(X_fit, y_fit)
print("predicting")
y_proba_ensemble = ensemble.predict_proba(X_test_proc[:10000])
thresh_ensemble = 0.35  # lower or higher to bias recall/precision
print("ensemble predicting")
y_pred_ensemble = np.where(y_proba_ensemble[:,1] >= thresh_ensemble, 1, np.argmax(y_proba_ensemble[:, [0,2]], axis=1) * 2)

print(classification_report(y_test[:10000], y_pred_ensemble[:10000], digits=4))


cm = confusion_matrix(y_test[:10000], y_pred_ensemble[:10000], labels=[0, 1, 2])   # order the labels as you like

# ── 2. plot it ───────────────────────────────────────────────────────────
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=['Low', 'Normal', 'High'])

fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(ax=ax, cmap='Blues', colorbar=True, values_format='d')  # any Matplotlib colormap works
ax.set_title("Confusion Matrix")
plt.tight_layout()
plt.show()"""
print(X_res.shape)

smote = SMOTE(random_state=42)
X_fit, y_fit = smote.fit_resample(X_fit, y_fit)
print(f"Resampled training dataset shape: {Counter(y_fit)}")

stack.fit(X_fit, y_fit)
print("here")

y_proba_stack = stack.predict_proba(X_test_proc)
print("here")
thresh_stack = 0.35  # lower or higher to bias recall/precision
y_pred_stack = np.where(y_proba_stack[:,1] >= thresh_stack, 1, np.argmax(y_proba_stack[:, [0,2]], axis=1) * 2)

print("here")
print(classification_report(y_test, y_pred_stack,    digits=4))

cm = confusion_matrix(y_test, y_pred_stack, labels=[0, 1, 2])   # order the labels as you like

# ── 2. plot it ───────────────────────────────────────────────────────────
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=['Low', 'Normal', 'High'])

fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(ax=ax, cmap='Blues', colorbar=True, values_format='d')  # any Matplotlib colormap works
ax.set_title("Confusion Matrix")
plt.tight_layout()
plt.show()