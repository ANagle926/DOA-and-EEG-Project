import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.ensemble import BalancedBaggingClassifier, EasyEnsembleClassifier
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE
from joblib import dump, load
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from imblearn.pipeline               import Pipeline as ImbPipeline
from VitalDBDataset import VitalDBDataset


pd.set_option('display.max_columns', None)

# stop wrapping into multiple blocks
pd.set_option('display.expand_frame_repr', False)


def process_data(x_train, x_test):

    surg_cols   = [
        "op_Biliary/Pancreas", "op_Breast", "op_Colorectal", "op_Hepatic",
        "op_Major resection", "op_Minor resection", "op_Others", "op_Stomach",
        "op_Thyroid", "op_Transplantation", "op_Vascular"
    ]

    numeric_feats  = ["gender", "bmi", "creatinine", "gpt"]
    categorical_feats = ["htn","dm","anemia"] + surg_cols

    all_cols = numeric_feats + categorical_feats

    df_train = pd.DataFrame(x_train, columns=all_cols)
    df_test  = pd.DataFrame(x_test,  columns=all_cols)


# Build preprocessor
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_feats),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_feats),
    ])

    return  preprocessor, df_train, df_test


x_train= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/x_train_preop3.joblib")
x_test= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/x_test_preop3.joblib")
y_train= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/y_train_preop3.joblib")
y_test= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/y_test_preop3.joblib")

preprocessor, df_x_train, df_x_test = process_data(x_train, x_test)

# 1.2. Base models
models = {
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
smotes = {
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

pipelines = {}
for name in models:
    pipelines[name] = ImbPipeline([
        ('pre',   preprocessor),
        ('smote', smotes[name]),
        ('clf',   models[name]),
    ])

# ── 4. Prepare OOF storage
n_classes = len(np.unique(y_train))
oof_probs = {name: np.zeros((len(y_train), n_classes))
             for name in pipelines}
y_true = y_train.copy()

# ── 5. 5‑fold CV loop
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, pipe in pipelines.items():
    for train_idx, val_idx in kf.split(df_x_train, y_train):
        # ← use .iloc to select rows by integer index
        X_tr, y_tr = df_x_train.iloc[train_idx], y_train[train_idx]
        X_val, y_val = df_x_train.iloc[val_idx],   y_train[val_idx]

        pipe.fit(X_tr, y_tr)
        oof_probs[name][val_idx] = pipe.predict_proba(X_val)

# ── 6. Derive hard labels
oof_preds = {name: oof_probs[name].argmax(axis=1)
             for name in pipelines}

# ── 7. Compute pairwise disagreement rate
def disagreement(a, b):
    return np.mean(a != b)

models_list = list(oof_preds)
m = len(models_list)
disc = pd.DataFrame(np.zeros((m, m)), index=models_list, columns=models_list)
for i, mi in enumerate(models_list):
    for j, mj in enumerate(models_list):
        disc.loc[mi, mj] = disagreement(oof_preds[mi], oof_preds[mj])

print("Disagreement rate:")
print(disc)

# ── 8. Compute error‐correlation
err = {name: (oof_preds[name] != y_true).astype(int) for name in pipelines}
err_mat = np.corrcoef([err[name] for name in models_list])
err_corr = pd.DataFrame(err_mat, index=models_list, columns=models_list)

print("\nError correlation:")
print(err_corr)
