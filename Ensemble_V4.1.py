from collections import Counter
from itertools import product
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import shap
from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.over_sampling import SMOTE
from joblib import dump, load
from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import StackingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, GridSearchCV, ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from VitalDBDataset import VitalDBDataset
"""vitaldb= VitalDBDataset(num_cases=5000, srate=128)

x_train= vitaldb.x_train
x_test= vitaldb.x_test
y_train= vitaldb.y_train
y_test= vitaldb.y_test
c_train= vitaldb.c_train
c_test= vitaldb.c_test

dump(x_train, "/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/x_train_preop2.joblib")
dump(x_test, "/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/x_test_preop2.joblib")
dump(y_train, "/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/y_train_preop2.joblib")
dump(y_test, "/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/y_test_preop2.joblib")
dump(c_train, "/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/c_train_preop2.joblib")
dump(c_test, "/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/c_test_preop2.joblib")"""


def process_data(x_train, x_test, y_train, y_test):

    y_train = np.where(y_train == 1, "normal",    "abnormal")
    y_test  = np.where(y_test  == 1, "normal",    "abnormal")


    surg_cols   = [
        "op_Biliary/Pancreas", "op_Breast", "op_Colorectal", "op_Hepatic",
        "op_Major resection", "op_Minor resection", "op_Others", "op_Stomach",
        "op_Thyroid", "op_Transplantation", "op_Vascular"
    ]

    all_cols = (["gender", "bmi"] + surg_cols + ["hypertension", "diabetes", "anemia", "creatine", "gpt"])

    df_train = (pd.DataFrame(x_train, columns=all_cols)
               #.drop(columns=surg_cols)
                )
    df_test  = (pd.DataFrame(x_test,  columns=all_cols)
                #.drop(columns=surg_cols)
                )

    continuous_feats = ["bmi", "creatine", "gpt"]
    binary_feats     = ["gender", "anemia", "hypertension", "diabetes"] + surg_cols

    preprocessor = ColumnTransformer([
        ("scale", StandardScaler(), continuous_feats),
        ("bin",   "passthrough",    binary_feats)
    ])

    x_train_proc = preprocessor.fit_transform(df_train)
    x_test_proc  = preprocessor.transform(df_test)

    print(f"Original training dataset shape: {Counter(y_train)}")
    X_res, y_res = SMOTE(random_state=42).fit_resample(x_train_proc, y_train)
    print(f"Resampled training dataset shape: {Counter(y_res)}")

    return X_res, y_res, x_test_proc, y_test, preprocessor

def run_permutation_importance(model, X, y,n_repeats=10, random_state=42, scoring='accuracy'):

    result = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring,
        n_jobs=-1
    )
    imp_df = pd.DataFrame({
        'feature': X.columns,
        'importance_mean': result.importances_mean,
        'importance_std':  result.importances_std,
    })

    imp_df = imp_df.sort_values('importance_mean', ascending=False).reset_index(drop=True)

    print(imp_df)
    plt.figure(figsize=(10,6))
    plt.bar(imp_df['feature'], imp_df['importance_mean'],
            yerr=imp_df['importance_std'], capsize=3)
    plt.xticks(rotation=90)
    plt.ylabel('Mean permutation importance')
    plt.title('Permutation Importances on Test Set')
    plt.tight_layout()
    plt.show()

def high_to_norm_error(stack, x_test, y_test, y_pred):

    # Identify misclassified High→Normal cases
    mask = (y_test == 2) & (y_pred == 1)
    X_mis = x_test[mask]

    explainer = shap.Explainer(stack.predict_proba, x_test)
    shap_vals_mis = explainer(X_mis)
    mean_shap_high = shap_vals_mis.values[:,:,2].mean(axis=0)

    mis_imp_df = pd.DataFrame({
        'feature': x_test.columns,
        'mean_shap_high': mean_shap_high
    }).sort_values('mean_shap_high', ascending=False)

    mis_imp_df['abs_shap'] = mis_imp_df['mean_shap_high'].abs()
    bad_by_mag = mis_imp_df[mis_imp_df['mean_shap_high'] < 0] \
        .sort_values('abs_shap', ascending=False) \
        .head(10)
    print(bad_by_mag[['feature','mean_shap_high']])

def norm_to_high_eror(stack, x_test, y_test, y_pred):

    # Identify misclassified Norm->High cases
    mask = (y_test == 1) & (y_pred == 2)
    X_mis = x_test[mask]

    explainer = shap.Explainer(stack.predict_proba, x_test)
    shap_vals_mis = explainer(X_mis)
    mean_shap_high = shap_vals_mis.values[:,:,1].mean(axis=0)

    mis_imp_df = pd.DataFrame({
        'feature': x_test.columns,
        'mean_shap_normal': mean_shap_high
    }).sort_values('mean_shap_normal', ascending=False)

    mis_imp_df['abs_shap'] = mis_imp_df['mean_shap_normal'].abs()
    bad_by_mag = mis_imp_df[mis_imp_df['mean_shap_normal'] < 0] \
        .sort_values('abs_shap', ascending=False) \
        .head(10)
    print(bad_by_mag[['feature','mean_shap_normal']])

def norm_to_low_eror(stack, x_test, y_test, y_pred):

    # Identify misclassified Norm->High cases
    mask = (y_test == 1) & (y_pred == 0)
    X_mis = x_test[mask]

    explainer = shap.Explainer(stack.predict_proba, x_test)
    shap_vals_mis = explainer(X_mis)
    mean_shap_high = shap_vals_mis.values[:,:,1].mean(axis=0)

    mis_imp_df = pd.DataFrame({
        'feature': x_test.columns,
        'mean_shap_normal': mean_shap_high
    }).sort_values('mean_shap_normal', ascending=False)

    mis_imp_df['abs_shap'] = mis_imp_df['mean_shap_normal'].abs()
    bad_by_mag = mis_imp_df[mis_imp_df['mean_shap_normal'] < 0] \
        .sort_values('abs_shap', ascending=False) \
        .head(10)
    print(bad_by_mag[['feature','mean_shap_normal']])

def low_to_norm_eror(stack, x_test, y_test, y_pred):

    #identify misclassified Low -> normal cases
    mask = (y_test == 0) & (y_pred == 1)
    X_mis = x_test[mask]

    explainer = shap.Explainer(stack.predict_proba, x_test)
    shap_vals_mis = explainer(X_mis)
    mean_shap_high = shap_vals_mis.values[:,:,0].mean(axis=0)

    mis_imp_df = pd.DataFrame({
        'feature': x_test.columns,
        'mean_shap_low': mean_shap_high
    }).sort_values('mean_shap_low', ascending=False)

    mis_imp_df['abs_shap'] = mis_imp_df['mean_shap_low'].abs()
    bad_by_mag = mis_imp_df[mis_imp_df['mean_shap_low'] < 0] \
        .sort_values('abs_shap', ascending=False) \
        .head(10)
    print(bad_by_mag[['feature','mean_shap_low']])



def optimise_thresholds(y_val: np.ndarray, proba_val: np.ndarray, *, step: float = 0.05) -> Tuple[float, float, Dict[str, float]]:
    """Return best (tau_low, tau_high) and a metrics dict."""
    best_f1 = -np.inf
    best_tau = (0.4, 0.4)
    metrics = {}
    taus = np.arange(step, 1.0, step)
    for tau_low, tau_high in product(taus, repeat=2):
        if tau_low >= tau_high:
            continue
        pred = _apply_thresholds(proba_val, tau_low, tau_high)
        f1 = f1_score(y_val, pred, average="macro")
        if f1 > best_f1:
            best_f1, best_tau = f1, (tau_low, tau_high)
    metrics["best_f1"] = best_f1
    return best_tau[0], best_tau[1], metrics

def tune_thresholds_cv(model, X, y, step=0.05, cv_splits=5, random_state=42):
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    oof_proba = cross_val_predict(model, X, y, cv=cv,
                                  method="predict_proba", n_jobs=-1)
    return optimise_thresholds(y, oof_proba, step=step)[:2]   # discard metrics

def _apply_thresholds(proba: np.ndarray, tau_low: float, tau_high: float) -> np.ndarray:
    pred = np.full(shape=(proba.shape[0],), fill_value=1, dtype=int)
    low_mask = proba[:, 0] >= tau_low
    high_mask = (proba[:, 2] >= tau_high) & (~low_mask)
    pred[high_mask] = 2
    pred[low_mask] = 0
    return pred



x_train= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/x_train_preop.joblib")
x_test= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/x_test_preop.joblib")
y_train= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/y_train_preop.joblib")
y_test= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/y_test_preop.joblib")

x_train_proc, y_train_proc, x_test_proc, y_test_proc, preprocessor= process_data(x_train, x_test, y_train, y_test)

feature_names = preprocessor.get_feature_names_out()

x_train_proc  = pd.DataFrame(x_train_proc, columns=feature_names)
x_test_proc = pd.DataFrame(x_test_proc, columns=feature_names)
#calibtrated svc
#qda
#knn
param_grid = {
    'n_neighbors': [5,10,15],
    'weights':     ['uniform','distance'],
    'algorithm':   ['auto','ball_tree','kd_tree','brute'],
    'leaf_size':   [10,20,30,40],
    'p':           [1,2]
}

for params in ParameterGrid(param_grid):
    knn = KNeighborsClassifier(**params)

    knn.fit(x_train_proc, y_train_proc)

    # 3c) Predict on the preprocessed test set
    y_pred = knn.predict(x_test_proc)

    # 3d) Compute the confusion matrix
    cm = confusion_matrix(y_test_proc, y_pred, labels=["abnormal","normal"])

    # 3e) Plot it
    disp = ConfusionMatrixDisplay(cm, display_labels=["abnormal","normal"])
    disp.plot(cmap="Blues", values_format='d')
    plt.title(f"KNN {params}")
    plt.tight_layout()
    plt.show()




"""qda = QuadraticDiscriminantAnalysis(reg_param=0.1)
ee= EasyEnsembleClassifier(n_estimators=50, random_state=42)

stack = StackingClassifier(
    estimators=[
        ('qda',  qda),
        ('ee', ee),
    ],
    final_estimator=LGBMClassifier(n_estimators=200, learning_rate=0.1, max_depth=  4,random_state=42, verbose=-1), #make this xgb?
    #make final estimator minority sensitive: BalancedRandomForestClassifie, LightGBM, CatBoost
    stack_method="predict_proba",
    cv=5,
    n_jobs=-1
)

stack.fit(x_train_proc, y_train_proc)
y_pred_num = stack.predict(x_test_proc)   # gives 0 or 1
y_pred = np.where(y_pred_num == 1, "normal", "abnormal")

print(classification_report(
    y_test_proc,
    y_pred,
    labels=["abnormal","normal"],
    target_names=["abnormal","normal"],
    zero_division=0
))

cm = confusion_matrix(y_test_proc, y_pred, labels=["abnormal","normal"])
fig, ax = plt.subplots(figsize=(6, 6))
ConfusionMatrixDisplay(cm, display_labels=["abnormal","normal"]).plot(ax=ax, cmap="Blues", values_format="d")
ax.set_title("Confusion Matrix – stack Set")
plt.tight_layout()
plt.show()"""

"""qda = QuadraticDiscriminantAnalysis(reg_param=0.1)

qda.fit(x_train_proc, y_train_proc)

#run_permutation_importance(qda, x_test_proc, y_test_proc, n_repeats=20)
probs = qda.predict_proba(x_test_proc)[:, 1]
#plt.hist(probs, bins=20)
#plt.title("Distribution of P(normal) on test set")
#plt.show()

thresh = 0.75
y_pred_custom = np.where(probs > thresh, "normal", "abnormal")
print(classification_report(y_test_proc, y_pred_custom))

cm = confusion_matrix(y_test_proc, y_pred_custom, labels=["abnormal","normal"])
fig, ax = plt.subplots(figsize=(6, 6))
ConfusionMatrixDisplay(cm, display_labels=["abnormal","normal"]).plot(ax=ax, cmap="Blues", values_format="d")
ax.set_title("Confusion Matrix – QDA Set")
plt.tight_layout()
plt.show()



lr = LogisticRegression(
    random_state=42,
    max_iter=1000,
    solver='lbfgs',
    class_weight='balanced'
)

lr.fit(x_train_proc, y_train_proc)

#run_permutation_importance(lr, x_test_proc, y_test_proc, n_repeats=20)
probs = lr.predict_proba(x_test_proc)[:, 1]   # P(normal)
#plt.hist(probs, bins=20)
#plt.title("Distribution of P(normal) on test set")
#plt.show()

#thresh = 0.87
#y_pred_custom = np.where(probs > thresh, "normal", "abnormal")
#print(classification_report(y_test_proc, y_pred_custom))
y_pred_num = lr.predict(x_test_proc)   # gives 0 or 1
y_pred = np.where(y_pred_num == 1, "normal", "abnormal")

print(classification_report(
    y_test_proc,
    y_pred,
    labels=["abnormal","normal"],
    target_names=["abnormal","normal"],
    zero_division=0
))


cm = confusion_matrix(y_test_proc, y_pred, labels=["abnormal","normal"])
fig, ax = plt.subplots(figsize=(6, 6))
ConfusionMatrixDisplay(cm, display_labels=["abnormal","normal"]).plot(ax=ax, cmap="Blues", values_format="d")
ax.set_title("Confusion Matrix – LR Set")
plt.tight_layout()
plt.show()"""
