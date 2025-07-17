from collections import Counter
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from joblib import load
from matplotlib import pyplot as plt
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE
from imblearn.ensemble    import EasyEnsembleClassifier, BalancedBaggingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree         import DecisionTreeClassifier
from sklearn.svm          import SVC
from sklearn.neighbors    import KNeighborsClassifier
from imblearn.pipeline               import Pipeline as ImbPipeline


#try without preprocessing?

x_train= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/x_train_preop3.joblib")
x_test= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/x_test_preop3.joblib")
y_train= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/y_train_preop3.joblib")
y_test= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/y_test_preop3.joblib")

# ————————————
# 1. Base models + per‑model SMOTE
# ————————————


knn_pipe = ImbPipeline([
    ('smote', BorderlineSMOTE(kind='borderline-2', random_state=42)),
    ('knn',  KNeighborsClassifier())
])

svc2_pipe = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('svc2',  SVC(probability=True, random_state=42))
])

sgd_pipe = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('sgd',   SGDClassifier(loss="log_loss", penalty="elasticnet",class_weight="balanced", random_state=42))
])

qda_pipe = ImbPipeline([
    ('smote', BorderlineSMOTE(kind="borderline-2", random_state=42)),
    ('qda',   QuadraticDiscriminantAnalysis(reg_param=0.01))
])

tabnet_pipe1 = ImbPipeline([
    ('smote', SVMSMOTE(sampling_strategy="not majority", random_state=42)),
    ('tabnet', TabNetClassifier(n_d=32, n_steps=5, gamma=1.5, seed=42, verbose=0))
])
tabnet_pipe2 = ImbPipeline([
    ('smote', SMOTEENN(sampling_strategy="not majority")),
    ('tabnet2', TabNetClassifier(n_d=32, n_steps=5, gamma=1.5, seed=42, verbose=0))
])

# ── 3. assemble the stack
estimators = [
    ('knn',    knn_pipe),
    ('svc2',   svc2_pipe),
    ('sgd',    sgd_pipe),
    ('qda',    qda_pipe),
    ('tabnet', tabnet_pipe1),
    ('tabnet2',tabnet_pipe2),
]


final_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=0.0)),
    ('lr',      LogisticRegression(
        penalty='l2',
        C=1.0,
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    ))
])

stack = StackingClassifier(
    estimators=estimators,       # your six pipelines
    final_estimator=final_pipe,  # <-- imputer + LR
    cv=5,
    n_jobs=-1,
    passthrough=False
)

stack.fit(x_train, y_train)
y_pred = stack.predict(x_test)

print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
classes = np.unique(y_test)   # or list of your class‐labels

# 3. Display it
cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
fig, ax = plt.subplots(figsize=(6, 6))
ConfusionMatrixDisplay(cm, display_labels=["Low", "Normal", "High"]).plot(ax=ax, cmap="Blues", values_format="d")
ax.set_title("Confusion Matrix – Test Set")
plt.tight_layout(); plt.show()

