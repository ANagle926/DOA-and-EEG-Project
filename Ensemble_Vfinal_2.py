import numpy as np
import pandas as pd
import shap
from imblearn.combine import SMOTEENN
from joblib import load, dump
from matplotlib import pyplot as plt
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, make_scorer, f1_score
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


#with preprocessing
import os, random
import numpy as np
import torch
from tensorflow.python.layers.core import dropout

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



def process_data(x_train, x_test):

    surg_cols   = [
        "op_Biliary/Pancreas", "op_Breast", "op_Colorectal", "op_Hepatic",
        "op_Major resection", "op_Minor resection", "op_Others", "op_Stomach",
        "op_Thyroid", "op_Transplantation", "op_Vascular"
    ]

    numeric_feats  = ["gender", "bmi", "creatinine", "gpt"]
    categorical_feats = ["htn","dm","anemia"]

    all_cols = numeric_feats + categorical_feats + surg_cols
    drop_cols = ["anemia"]

    df_train = (
        pd.DataFrame(x_train, columns=all_cols)
        .drop(columns=drop_cols)
    )
    df_test  = (
        pd.DataFrame(x_test,  columns=all_cols)
        .drop(columns=drop_cols)
    )

    # Build preprocessor
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(),           ["gender","bmi","creatinine"]),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["dm", "htn"]),
    ])

    return  preprocessor, df_train, df_test

def create_model(preprocessor):

    weights = {0: 10,
               1: 3, #og: 1 is 1 and 2 is 2
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

    estimators = [
        ('knn',    knn_pipe),
        ('sgd',    sgd_pipe),
        ('bb', bb_pipe),
        ('tabnet', tabnet_pipe1),
        ('tabnet2',tabnet_pipe2),
    ]

    final_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0.0)),
        ('lr',      LogisticRegression(
            penalty='l2',
            C=1.0,
            class_weight=weights,
            max_iter=1000,
            random_state=42,
        ))
    ])

    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=final_pipe,
        cv=StratifiedKFold(
            n_splits=5,
            shuffle=False    # no shuffling = deterministic splits
        ),
        n_jobs=1,            # single‑threaded for reproducibility
        passthrough=False
    )

    return stack

def apply_thresholds(y_proba):

    TAU_LOW  = 0.4
    TAU_NORMAL = 0.35
    TAU_HIGH = 0.3

    pred = np.full(shape=(y_proba.shape[0],), fill_value=1, dtype=int)

    norm = (y_proba[:, 1] >= TAU_NORMAL)
    high = (y_proba[:, 2] >= TAU_HIGH) & (~norm)
    low  = (y_proba[:, 0] >= TAU_LOW)  & (~norm) & (~high)

    pred[high] = 2
    pred[low]  = 0
    pred[norm] = 1

    return pred

def train_and_evaluate(x_train, x_test, y_train, y_test, preprocessor):

    stack = create_model(preprocessor)
    stack.fit(x_train, y_train)
    dump(stack, "stacked_ensemble3")

    stack = load("stacked_ensemble3")

    #run_permutation_importance(stack, x_test, y_test, n_repeats=20)
    y_pred_proba = stack.predict_proba(x_test)
    y_pred= apply_thresholds(y_pred_proba)

    #y_pred = stack.predict(x_test)

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

#dump(preprocessor, "preprocessor.joblib")
#dump(df_x_train, "df_x_train.joblib")
#dump(df_x_test, "df_x_test.joblib")

#df_x_train= load("df_x_train.joblib")
#df_x_test= load("df_x_test.joblib")
#preprocessor= load("preprocessor.joblib")

stack, y_pred = train_and_evaluate(df_x_train, df_x_test, y_train, y_test, preprocessor)

print("high to norm")
high_to_norm_error(stack, df_x_test, y_test, y_pred)
print("norm to high")
norm_to_high_eror(stack, df_x_test, y_test, y_pred)
print("norm to low")
norm_to_low_eror(stack, df_x_test, y_test, y_pred)
print("low to norm")
low_to_norm_eror(stack, df_x_test, y_test, y_pred)
