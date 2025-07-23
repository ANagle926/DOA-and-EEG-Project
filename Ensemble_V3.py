from typing import Tuple, Dict

import pandas as pd
import shap
from imblearn.combine import SMOTEENN
from joblib import load, dump
from matplotlib import pyplot as plt
from ngboost import NGBClassifier
from ngboost.distns import k_categorical
from ngboost.scores import CRPS, CRPScore
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import StackingClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, \
    average_precision_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE
from imblearn.ensemble    import  BalancedBaggingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree         import DecisionTreeClassifier
from sklearn.neighbors    import KNeighborsClassifier
from imblearn.pipeline               import Pipeline as ImbPipeline
import os, random
import numpy as np
import torch
from itertools import product     # add this import
#from mord import OrdinalLogistic

os.environ['PYTHONHASHSEED']     = '42'
random.seed(42)
np.random.seed(42)

# 1b) Single‑thread your linear‑algebra libs
os.environ['OMP_NUM_THREADS']     = '1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']     = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# 1c) Torch RNG
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# 1d) Force deterministic backends
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
try:
    torch.use_deterministic_algorithms(True)
except:
    pass

def plot_pr_curves(model, X_val: pd.DataFrame, y_val: np.ndarray, *, ax=None) -> None:
    """Plot one‑vs‑rest PR curves for classes 0, 1, 2."""

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    proba = model.predict_proba(X_val)
    for cls in [0, 1, 2]:
        prec, rec, _ = precision_recall_curve((y_val == cls).astype(int), proba[:, cls])
        ap = average_precision_score((y_val == cls).astype(int), proba[:, cls])
        ax.plot(rec, prec, label=f"Class {cls} (AP={ap:.2f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("One‑vs‑Rest Precision‑Recall Curves")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

def process_data(x_train, x_test):

    surg_cols   = [
        "op_Biliary/Pancreas", "op_Breast", "op_Colorectal", "op_Hepatic",
        "op_Major resection", "op_Minor resection", "op_Others", "op_Stomach",
        "op_Thyroid", "op_Transplantation", "op_Vascular"
    ]

    numeric_feats  = ["gender", "bmi", "creatinine", "gpt"]
    categorical_feats = ["htn","dm","anemia"]

    all_cols = numeric_feats + categorical_feats  + surg_cols
    #drop_cols = surg_cols + ["gender"]

    df_train = pd.DataFrame(x_train, columns=all_cols)
    df_test  = pd.DataFrame(x_test,  columns=all_cols)

    for df in (df_train, df_test):
        # surgery aggregates FIRST
        df["major_resection"] = ((df["op_Major resection"] == 1) |
                                 (df["op_Transplantation"] == 1)).astype(int)
        df["pancreatic_surg"] = df["op_Biliary/Pancreas"].astype(int)
        df["vascular_surg"]   = df["op_Vascular"].astype(int)

        # BMI buckets & GPT quartile
        df["BMI_under"]  = (df["bmi"] < 18.5).astype(int)
        df["BMI_normal"] = df["bmi"].between(18.5, 25, inclusive="left").astype(int)
        df["BMI_over"]   = (df["bmi"] >= 25).astype(int)
        df["gpt_q"]      = pd.qcut(df["gpt"], q=4, labels=False, duplicates="drop")

    # now drop individual surgery dummies
    #df_train.drop(columns=drop_cols, inplace=True)
    #df_test.drop(columns=drop_cols, inplace=True)

    num_out = ["bmi", "creatinine", "gpt", "gender"] + ["gpt_q"]
    cat_out = categorical_feats + ["BMI_under", "BMI_over", "major_resection", "pancreatic_surg", "vascular_surg", "BMI_normal"] + surg_cols

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_out),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_out),
    ])

    return  preprocessor, df_train, df_test

def create_model(preprocessor):

    weights = {0: 10,
               1: 3,
               2: 2}

    knn_pipe = ImbPipeline([
        ('pre',   preprocessor),
        ('smote', BorderlineSMOTE(kind='borderline-2', random_state=42)),
        ('knn',  KNeighborsClassifier())
    ])
    sgd_pipe = ImbPipeline([
        ('pre',   preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('sgd',   SGDClassifier(loss="log_loss", penalty="elasticnet",class_weight=weights, random_state=42))
    ])
    bb_pipe = ImbPipeline([
        ('pre',   preprocessor),
        ('smote', SMOTEENN(sampling_strategy="not majority", random_state=42)),
        ('bb', BalancedBaggingClassifier(estimator=DecisionTreeClassifier(max_depth=6), sampling_strategy="auto", n_estimators=10,max_samples=0.5, replacement=True, random_state=42, n_jobs=1))
    ])
    tabnet_pipe1 = ImbPipeline([
        ('pre',   preprocessor),
        ('smote', SVMSMOTE(sampling_strategy="not majority", random_state=42)),
        ('tabnet', TabNetClassifier(n_d=32, n_steps=5, gamma=1.5, seed=42, verbose=0, device_name='cpu'))
    ])
    tabnet_pipe2 = ImbPipeline([
        ('pre',   preprocessor),
        ('smote', SMOTEENN(sampling_strategy="not majority", random_state=42)),
        ('tabnet2', TabNetClassifier(n_d=32, n_steps=5, gamma=1.5, seed=42, verbose=0, device_name='cpu'))
    ])

    """ordinal_pipe = ImbPipeline([
        ('pre',   preprocessor),
        ('smote', KMeansSMOTE(random_state=42)),        # cleaner synth. pts.
        ('ord',   OrdinalLogistic(alpha=1.0)),
    ])"""

    """ngb_pipe = ImbPipeline([
        ('pre',   preprocessor),
        ('ngb',   NGBClassifier(
            Dist=k_categorical(3),
            Score=CRPScore,                      # <- drop the () here
            natural_gradient=True,
            minibatch_frac=1.0,
            random_state=42))
    ])"""

    estimators = [
        ('knn',    knn_pipe),
        ('sgd',    sgd_pipe),
        ('bb', bb_pipe),
        ('tabnet', tabnet_pipe1),
        ('tabnet2',tabnet_pipe2),
        #('ord', ordinal_pipe),
        #('ngb', ngb_pipe)
    ]

    final_pipe = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.1, random_state=42)

    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=final_pipe,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=1,
        passthrough=False
    )

    return stack

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


def train_and_evaluate(x_train, x_test, y_train, y_test, preprocessor):

    stack = create_model(preprocessor)
    print("beginning threshold search")
    tau_low, tau_high = tune_thresholds_cv(stack, x_train, y_train)
    print("tau_low is", tau_low)
    print("tau_high is", tau_high)

    stack.fit(x_train, y_train)
    dump(stack, "stacked_ensemble5")

    print("finished fitting")
    y_pred = _apply_thresholds(stack.predict_proba(x_test), tau_low, tau_high)
    plot_pr_curves(stack, x_train, y_train)

    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay(cm, display_labels=["Low", "Normal", "High"]).plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Confusion Matrix – Test Set")
    plt.tight_layout()
    plt.show()

    return stack, y_pred


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

x_train= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/x_train_preop3.joblib")
x_test= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/x_test_preop3.joblib")
y_train= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/y_train_preop3.joblib")
y_test= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/y_test_preop3.joblib")

preprocessor, df_x_train, df_x_test = process_data(x_train, x_test)

stack, y_pred = train_and_evaluate(df_x_train, df_x_test, y_train, y_test, preprocessor)

run_permutation_importance(stack, df_x_test, y_test, n_repeats=20)
print("high to norm")
high_to_norm_error(stack, df_x_test, y_test, y_pred)
print("norm to high")
norm_to_high_eror(stack, df_x_test, y_test, y_pred)
print("norm to low")
norm_to_low_eror(stack, df_x_test, y_test, y_pred)
print("low to norm")
low_to_norm_eror(stack, df_x_test, y_test, y_pred)
