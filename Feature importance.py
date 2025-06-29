import json
from collections import Counter
import seaborn as sns
import joblib
import shap
import numpy as np
import pandas as pd
from IPython.core.display_functions import display
from catboost import CatBoostClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.over_sampling import SMOTE
from joblib import load, dump
from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from xgboost import XGBClassifier

from VitalDBDataset import VitalDBDataset

"""dataset= VitalDBDataset(num_cases=500)
x_train= dataset.x_train
x_test= dataset.x_test
y_train= dataset.y_train
y_test= dataset.y_test
c_train= dataset.c_train
c_test= dataset.c_test

dump(x_train, "/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/x_train_preop.joblib")
dump(x_test, "/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/x_test_preop.joblib")
dump(y_train, "/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/y_train_preop.joblib")
dump(y_test, "/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/y_test_preop.joblib")
dump(c_train, "/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/c_train_preop.joblib")
dump(c_test, "/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/c_test_preop.joblib")"""

def bootstrap_metric(y_true, y_pred, metric_func, n_bootstrap=1000, random_state=42):
    np.random.seed(random_state)
    bootstrapped_scores = []
    y_true = np.array(y_true)
    y_pred  = np.array(y_pred)

    for _ in range(n_bootstrap):
        # sample with replacement
        idx = np.random.choice(len(y_true), size=len(y_true), replace=True)
        yt = y_true[idx]
        yp = y_pred[idx]

        # for 1D predictions, need at least two classes
        if yp.ndim == 1:
            if len(np.unique(yt)) < 2:
                continue

        # for multiclass probabilities, need all classes present
        else:
            n_classes = yp.shape[1]
            if len(np.unique(yt)) != n_classes:
                continue

        try:
            score = metric_func(yt, yp)
        except ValueError:
            # e.g. AUC on mismatched classes
            continue

        bootstrapped_scores.append(score)

    if not bootstrapped_scores:
        return np.nan, np.nan

    return (
        np.percentile(bootstrapped_scores, 2.5),
        np.percentile(bootstrapped_scores, 97.5)
    )

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["low","normal","high"],
                yticklabels=["low","normal","high"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix for {model_name}")
    plt.show()

x_train= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/x_train_preop.joblib")
x_test= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/x_test_preop.joblib")
y_train= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/y_train_preop.joblib")
y_test= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/y_test_preop.joblib")

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
    ("cat", OneHotEncoder(drop="first"),     categorical_feats),
])

# Convert x_train/x_test into DataFrames
df_train = pd.DataFrame(x_train, columns=all_cols)
df_test  = pd.DataFrame(x_test,  columns=all_cols)

df_all = pd.concat([df_train, df_test], axis=0, ignore_index=True)
preprocessor.fit(df_all)

X_train_proc = preprocessor.transform(df_train)
X_test_proc  = preprocessor.transform(df_test)


"""# SMOTE which over-samples the training set
print(f"Original training dataset shape: {Counter(y_train)}")
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train_proc, y_train)
print(f"Resampled training dataset shape: {Counter(y_res)}")

dump(X_res, "/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/x_smote_preop.joblib")
dump(y_res, "/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/y_smote_preop.joblib")"""

X_res= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/x_smote_preop.joblib")
y_res= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/y_smote_preop.joblib")


"""# 1. Fit your LR
lr = LogisticRegression(
    class_weight='balanced',
    solver='lbfgs',
    max_iter=5000,
    random_state=42
)
lr.fit(X_res, y_res)

# 2. Recover feature names
cat_ohe_names = preprocessor.named_transformers_['cat'] \
    .get_feature_names_out(categorical_feats) \
    .tolist()

feat_names = numeric_feats + cat_ohe_names
assert len(feat_names) == X_train_proc.shape[1]
assert len(feat_names) == X_test_proc.shape[1]
dump(feat_names, "/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/feat_names.joblib")


# 3. Pull out and sort coefficients

#coefs= lr.coef_[0] will say which features help predict class 0 (minority)

coefs = np.sum(np.abs(lr.coef_), axis=0)
df_imp = (
    pd.DataFrame({'feature': feat_names, 'coef': coefs})
    .assign(abs_coef=lambda d: d.coef.abs())
    .sort_values('abs_coef', ascending=False)
    .reset_index(drop=True)
)

# 4. Display top positive & negative drivers
print("Top features pushing toward higher‐MAC class:")
display(df_imp.nlargest(5, 'coef')[['feature','coef']])

print("Top features pushing toward lower‐MAC class:")
display(df_imp.nsmallest(5, 'coef')[['feature','coef']])


print("→ now running permutation_importance, this may take a while …")

X_test_dense = X_test_proc.toarray()
perm = permutation_importance(
    lr,
    X_test_dense,
    y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)


df_perm = (
    pd.DataFrame({'feature': feat_names,'importance': perm.importances_mean})
    .sort_values('importance', ascending=False)
    .reset_index(drop=True)
)
print("Top permutation importances for LR:")
display(df_perm.head(10))

top10 = df_perm.head(10).iloc[::-1]

plt.figure(figsize=(8,6))
plt.barh(top10['feature'], top10['importance'], edgecolor='k')
plt.xlabel("Permutation importance")
plt.title("Top 10 Features (Permutation Importance)")
plt.tight_layout()
plt.show()

top_coefs = df_imp.head(10).iloc[::-1]  # biggest absolute coefficients

plt.figure(figsize=(8,6))
plt.barh(top_coefs['feature'], top_coefs['coef'], color=(top_coefs['coef']>0).map({True:'C0', False:'C3'}))
plt.xlabel("Logistic Regression coefficient")
plt.title("Top 10 Features (Model Coefficients)")
plt.axvline(0, color='gray', linestyle='--')
plt.tight_layout()
plt.show()
print("here")
masker = shap.maskers.Independent(X_train_proc, max_samples=2000)
explainer = shap.Explainer(lr, masker, feature_names=feat_names)
shap_values = explainer(X_test_proc[:4000])
print("here")
shap.plots.beeswarm(shap_values[:, :, 0], max_display=20)
shap.plots.beeswarm(shap_values[:, :, 1], max_display=20)
shap.plots.beeswarm(shap_values[:, :, 2], max_display=20)"""
X_res_dense = X_res.toarray()
X_test_proc= X_test_proc.toarray()

models = {
    """"'Logistic Regression': LogisticRegression(random_state=42),
    'Ridge Classifier': RidgeClassifier(),
    'MLP Classifier': MLPClassifier(random_state=42, max_iter=300),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Extra Trees': ExtraTreesClassifier(random_state=42),
    'Decision Trees': DecisionTreeClassifier(random_state=42),
    "XGB": XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42, use_label_encoder=False, eval_metric="mlogloss"),
    "CalibratedSVC": CalibratedClassifierCV(LinearSVC(max_iter=5000), cv=3),
    "LightGBM": LGBMClassifier(n_estimators=200, learning_rate=0.1, max_depth=  4, random_state=42),
    "CatBoost": CatBoostClassifier(iterations=200, learning_rate=0.1, depth=6, random_seed=42, verbose=False),
    "BalancedBagging": BalancedBaggingClassifier(estimator=DecisionTreeClassifier(max_depth=6), sampling_strategy="auto", n_estimators=10,max_samples=0.5, replacement=True, random_state=42, n_jobs=1),
    "RandomForestBalanced": RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced", random_state=42,n_jobs=-1),"""
    "GNB": GaussianNB(),
    "LDA": LinearDiscriminantAnalysis()
}

# Compare model metrics with a great table
results = []
print("here")
for model_name, model in tqdm(models.items(), desc="Training and Evaluating Models"):
    print("here")
    model.fit(X_res_dense, y_res)
    print("here")
    y_pred = model.predict(X_test_proc)
    y_pred_proba = model.predict_proba(X_test_proc) if hasattr(model, "predict_proba") else None
    print("here")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test,   y_pred, average="macro",zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba, multi_class="ovr") if y_pred_proba is not None else np.nan

    # Bootstrapped confidence intervals
    acc_ci = bootstrap_metric(y_test, y_pred, accuracy_score)
    prec_ci = bootstrap_metric(
        y_test, y_pred,
        lambda yt, yp: precision_score(yt, yp, average="macro", zero_division=1)
    )
    recall_ci = bootstrap_metric(
        y_test, y_pred,
        lambda yt, yp: recall_score(yt, yp, average="macro")
    )
    f1_ci = bootstrap_metric(
        y_test, y_pred,
        lambda yt, yp: f1_score(yt, yp, average="macro")
    )
    auc_ci   = (bootstrap_metric(y_test, y_pred_proba, lambda y, p: roc_auc_score(y, p, multi_class="ovr")) if y_pred_proba is not None else (np.nan, np.nan))

    # Storing
    results.append({
        "Model": model_name,
        "Accuracy": f"{accuracy:.4f} ({acc_ci[0]:.4f}, {acc_ci[1]:.4f})",
        "Precision": f"{precision:.4f} ({prec_ci[0]:.4f}, {prec_ci[1]:.4f})",
        "Recall": f"{recall:.4f} ({recall_ci[0]:.4f}, {recall_ci[1]:.4f})",
        "F1-Score": f"{f1:.4f} ({f1_ci[0]:.4f}, {f1_ci[1]:.4f})",
        "AUC": f"{auc:.4f} ({auc_ci[0]:.4f}, {auc_ci[1]:.4f})" if y_pred_proba is not None else "N/A"
    })
    plot_confusion_matrix(y_test, y_pred, model_name)

# Display as a styled table
results_df = pd.DataFrame(results)
styled_results_df = results_df.style.set_table_styles(
    [{'selector': 'th', 'props': [('font-weight', 'bold'), ('background-color', '#f0f0f0')]},
     {'selector': 'td', 'props': [('padding', '5px')]}]
).set_properties(**{'text-align': 'center'}).set_caption("Model Evaluation Results with SMOTE")

display(styled_results_df)
