import json
from collections import Counter
import seaborn as sns
import joblib
import shap
import numpy as np
import pandas as pd
from IPython.core.display_functions import display
from imblearn.ensemble import BalancedBaggingClassifier, BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE
from joblib import load, dump
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance
from imblearn.ensemble import BalancedBaggingClassifier

"""x_train= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/x_train_preop.joblib")
x_test = load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/x_test_preop.joblib")
y_train= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/y_train_preop.joblib")
y_test = load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/y_test_preop.joblib")

static_cols = ["gender", "bmi", "htn", "dm", "anemia", "ph", "creatinine", "gpt", "paO2", "paCO2"]
surg_cols   = [
    "op_Biliary/Pancreas", "op_Breast", "op_Colorectal", "op_Hepatic",
    "op_Major resection", "op_Minor resection", "op_Others", "op_Stomach",
    "op_Thyroid", "op_Transplantation", "op_Vascular"
]

all_cols = static_cols + surg_cols

numeric_feats  = ["bmi", "ph", "creatinine", "gpt", "paO2", "paCO2"]
categorical_feats = ["gender","htn","dm","anemia"] + surg_cols


binary_levels = [[0, 1]] * len(categorical_feats)   # one [0,1] list per feature

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_feats),
    ("cat", OneHotEncoder(
        categories=binary_levels,   # keeps column count fixed
        drop=None,                  # <— keep BOTH dummies
        handle_unknown="ignore",    # unseen categories → all-zero row
        dtype=int                   # saves memory vs float
    ), categorical_feats),
])

# Convert x_train/x_test into DataFrames
df_train = pd.DataFrame(x_train, columns=all_cols)
df_test  = pd.DataFrame(x_test,  columns=all_cols)

preprocessor.fit(df_train)

X_train_proc = preprocessor.transform(df_train)
X_test_proc  = preprocessor.transform(df_test)

print(f"Original training dataset shape: {Counter(y_train)}")
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train_proc, y_train)
print(f"Resampled training dataset shape: {Counter(y_res)}")

dump(X_res, "/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/x_smote_preop.joblib")
dump(y_res, "/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/y_smote_preop.joblib")
dump(preprocessor, "/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/preproc.joblib")
dump(X_train_proc, "/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/x_train_proc.joblib")
dump(X_test_proc, "/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/x_test_proc.joblib")"""

preprocessor= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/preproc.joblib")
X_test_proc= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/x_test_proc.joblib")
X_train_proc= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/x_train_proc.joblib")
X_res= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/x_smote_preop.joblib")
y_res= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/y_smote_preop.joblib")



""""# 1. Fit your LR
bb = BalancedBaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=6),
    sampling_strategy="auto",
    n_estimators=10,
    max_samples=0.5,
    replacement=True,
    random_state=42,
    n_jobs=1
)
bb.fit(X_res, y_res)

# 2. Recover feature names
cat_ohe_names = preprocessor.named_transformers_['cat'] \
    .get_feature_names_out(categorical_feats) \
    .tolist()

feat_names = numeric_feats + cat_ohe_names
assert len(feat_names) == X_train_proc.shape[1]
assert len(feat_names) == X_test_proc.shape[1]
dump(feat_names, "/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/feat_names.joblib")


# 3. Pull out and sort coefficients
print("→ now running permutation_importance, this may take a while …")

X_test_dense = X_test_proc.toarray()
rng = np.random.RandomState(42)
idx = rng.choice(len(y_test), size=5000, replace=False)

X_sub = X_test_dense[idx]
y_sub = y_test[idx]


print("here")
perm = permutation_importance(
    bb,
    X_sub,
    y_sub,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

print("here")
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




# 1) sample indices properly
rng   = np.random.RandomState(42)
n_rows = X_test_proc.shape[0]
idx   = rng.choice(n_rows, size=5000, replace=False)

# 2) extract and densify if needed
X_sub = X_test_proc[idx]
X_sub_dense = X_sub.toarray()

# 3) explain
model_fn  = lambda X: bb.predict_proba(X)
explainer = shap.Explainer(model_fn, X_train_proc, feature_names=feat_names)
shap_values = explainer(X_sub_dense)

# 4) plot per class
for cls in range(shap_values.shape[-1]):
    shap.plots.beeswarm(shap_values[..., cls], max_display=20, show=False)
    plt.title(f"SHAP Beeswarm for class {cls}")
    plt.tight_layout()
    plt.show()"""

"""bb_clf = BalancedBaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=None, random_state=42),
    n_estimators=150,
    sampling_strategy="auto",    # not used here – we already balanced w/ SMOTE
    n_jobs=-1,
    random_state=42,
)

bb_clf.fit(X_res, y_res)

# Quick hold‑out performance --------------------------------------------------
n_classes = len(np.unique(y_res))

if n_classes == 2:
    proba_test = bb_clf.predict_proba(X_test_proc)[:, 1]
    auc = roc_auc_score(y_test, proba_test)
    label = "binary"
else:
    proba_test = bb_clf.predict_proba(X_test_proc)
    auc = roc_auc_score(
        y_test, proba_test, multi_class="ovr", average="macro"
    )
    label = f"{n_classes}-class OVR‑macro"

print(f"Hold‑out AUROC ({label}): {auc:.3f}")

# ----------------------------------------------------------------------------
# 2. SHAP explanation (model‑agnostic; SHAP picks TreeExplainer internally)
# ----------------------------------------------------------------------------

background = shap.sample(X_res, 100, random_state=42)
explainer = shap.PermutationExplainer(bb_clf.predict_proba, background)
shap_values = explainer(X_test_proc)

plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values.values if hasattr(shap_values, "values") else shap_values,
    features=X_test_proc,
    feature_names=all_cols,
    show=False,
)
plt.tight_layout()
plt.savefig("shap_beeswarm_balancedbagging.png", dpi=200)
plt.close()
print("Saved SHAP beeswarm plot → shap_beeswarm_balancedbagging.png")

# ----------------------------------------------------------------------------
# 3. Permutation importance (AUROC drop) as complementary sanity‑check
# ----------------------------------------------------------------------------

perm = permutation_importance(
    bb_clf,
    X_test_proc,
    y_test,
    n_repeats=15,
    random_state=42,
    n_jobs=-1,
)

perm_series = pd.Series(perm.importances_mean, index=all_cols).sort_values(ascending=False)

plt.figure(figsize=(8, 10))
perm_series.iloc[:30][::-1].plot(kind="barh")  # top‑30 most important
plt.xlabel("Mean AUROC decrease (perm)")
plt.title("Permutation Feature Importance – BalancedBagging (SMOTE)")
plt.tight_layout()
plt.savefig("perm_importance_balancedbagging_smote.png", dpi=200)
plt.close()
print("Saved permutation bar chart -> perm_importance_balancedbagging_smote.png")

# ----------------------------------------------------------------------------
# 4. Persist artifacts for future analysis
# ----------------------------------------------------------------------------

joblib.dump(bb_clf, "balancedbagging_smote.joblib")
explainer.save("shap_explainer_balancedbagging_smote.pkl")
print("Artifacts saved (model + SHAP explainer)")"""

"""bbc = BalancedBaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=200,
    sampling_strategy="auto",
    replacement=False,
    oob_score=True,          # <— if you want that metric
    random_state=42,
    n_jobs=-1
)"""
bbc = BalancedRandomForestClassifier(
    n_estimators=300,          # lots of trees but still fast
    max_depth=None,
    n_jobs=-1,
    bootstrap=True,
    oob_score=True,          # ← so .oob_score_ is defined
    random_state=42
)

bbc.fit(X_res, y_res)
print("Balanced Bagging OOB-score:", bbc.oob_score_)

import shap, functools, scipy.sparse as sp


X_train_dense = X_res.toarray()      if sp.issparse(X_res)      else X_res
X_test_dense  = X_test_proc.toarray() if sp.issparse(X_test_proc) else X_test_proc

"""masker = shap.maskers.Independent(X_train_dense, max_samples=1000)

predict_fn = functools.partial(bbc.predict_proba)

explainer = shap.Explainer(
    predict_fn,
    masker,
    algorithm="permutation",        # force permutation so you know what you’re getting
    output_names=bbc.classes_
)
print("here")
# --- ❹ get SHAP values ---
shap_values = explainer(X_test_dense, max_evals="auto")
print("almost at loop")"""

feat_names = preprocessor.get_feature_names_out()
explainer  = shap.TreeExplainer(bbc, X_train_dense, feature_names=feat_names)
shap_vals  = explainer.shap_values(X_test_dense)      # seconds, not hours!



for c_idx, c_lab in enumerate(bbc.classes_):
    shap.summary_plot(
        shap_vals.values[:, :, c_idx],      # pull the array for one output
        X_test_dense,
        feature_names=preprocessor.get_feature_names_out(),
        show=False
    )
    plt.title(f"Balanced Bagging – class {c_lab}")
    plt.tight_layout()
    plt.show()
