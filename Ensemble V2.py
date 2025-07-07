from collections import Counter
from hashlib import md5

import numpy as np
import pandas as pd
import shap
from IPython.core.display_functions import display
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
feat_names= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/feat_names.joblib")

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
    final_estimator=LogisticRegression(max_iter=10_000, class_weight="balanced"),
    stack_method="predict_proba",
    cv=5,
    n_jobs=-1
)

from sklearn.model_selection import train_test_split, GroupShuffleSplit
print(X_res.shape)
print(y_res.shape)

X_fit, _, y_fit, _ = train_test_split(
    X_res, y_res,
    train_size=5000,
    stratify=y_res,
    random_state=42
)

print(X_fit.shape)
smote = SMOTE(random_state=42)
X_fit, y_fit = smote.fit_resample(X_fit, y_fit)
print(f"Resampled training dataset shape: {Counter(y_fit)}")

stack.fit(X_fit, y_fit)


y_proba_stack = stack.predict_proba(X_test_proc)
thresh_stack = 0.1
print("with threshold of", thresh_stack)
y_pred_stack = np.where(y_proba_stack[:,1] >= thresh_stack, 1, np.argmax(y_proba_stack[:, [0,2]], axis=1) * 2)
print(classification_report(y_test, y_pred_stack, digits=4))

cm = confusion_matrix(y_test, y_pred_stack, labels=[0, 1, 2])   # order the labels as you like

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=['Low', 'Normal', 'High'])

fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(ax=ax, cmap='Blues', colorbar=True, values_format='d')  # any Matplotlib colormap works
ax.set_title("Confusion Matrix")
plt.tight_layout()
plt.show()

#feat_names = preprocessor.get_feature_names_out(all_cols)
#dump(feat_names, "/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/feat_names.joblib")



mask_high = y_test == 2
y_err = (y_pred_stack[mask_high] == 1).astype(int)
X_high = X_test_proc[mask_high]

X_tr, X_va, y_tr, y_va = train_test_split(X_high, y_err, test_size=0.3,
                                          stratify=y_err, random_state=42)
perfect_cols = [c for c in feat_names if
                (X_tr[:, feat_names == c] == y_tr.reshape(-1,1)).all()]
print("Leaks:", perfect_cols)
logi = LogisticRegression(max_iter=5000, penalty='l1', solver='liblinear')
logi.fit(X_tr, y_tr)
print(classification_report(y_va, logi.predict(X_va)))



coef = pd.Series(logi.coef_.ravel(), index=feat_names)
print(coef.sort_values(ascending=False).head(15))
stump = DecisionTreeClassifier(max_depth=1).fit(X_tr, y_tr)
print(stump.score(X_va, y_va))
df = pd.DataFrame(X_high, columns=feat_names)
corr = df.corrwith(pd.Series(y_err, name='err')).abs().sort_values(ascending=False)
print(corr.head(10))


row_hash = np.array([md5(r.tobytes()).hexdigest() for r in X_high])
gss = GroupShuffleSplit(test_size=0.3, n_splits=1, random_state=42)
train_idx, val_idx = next(gss.split(X_high, y_err, groups=row_hash))
X_tr, X_va = X_high[train_idx], X_high[val_idx]
y_tr, y_va = y_err[train_idx], y_err[val_idx]
logi = LogisticRegression(max_iter=5000, penalty='l1', solver='liblinear')
logi.fit(X_tr, y_tr)
print(classification_report(y_va, logi.predict(X_va)))
stump = DecisionTreeClassifier(max_depth=1).fit(X_tr, y_tr)
print("stump acc:", stump.score(X_va, y_va))


mis02 = (y_test == 2) & (y_pred_stack == 1)   # true High → predicted Normal
mis01 = (y_test == 0) & (y_pred_stack == 1)   # true Low  → predicted Normal

X_high2norm = X_test_proc[mis02]
X_low2norm  = X_test_proc[mis01]
print(f"High→Norm rows: {X_high2norm.shape[0]}")
print(f"Low →Norm rows: {X_low2norm.shape[0]}")

xgb_fitted = stack.named_estimators_['xgb']
explainer = shap.TreeExplainer(xgb_fitted)   # xgb is the base learner object

rows = min(2000, X_high2norm.shape[0])
sh_high2norm = explainer(X_high2norm[:rows], check_additivity=False)
rows = min(2000, X_low2norm.shape[0])
sh_low2norm  = explainer(X_low2norm[:rows],  check_additivity=False)

def top_features(sh_values, cls=1, k=15):
    """
    sh_values : shap.Explanation OR numpy array
        For a multiclass model this is shape (C, n, p).
    cls       : which class’s SHAP values to aggregate (Normal = 1).
    """
    vals = sh_values.values
    if vals.ndim == 3:          # C × n × p
        vals = vals[cls]        # take the class you care about
    imp   = np.abs(vals).mean(axis=0)
    order = np.argsort(imp)[::-1][:k]
    return pd.DataFrame({
        "feature": np.array(feat_names)[order],
        "mean_abs_shap": imp[order]
    })

top_high = top_features(sh_high2norm, k=20)
top_low  = top_features(sh_low2norm, k=20)

print("True High → predicted Normal:")
print(top_high.head(10))

print("\nTrue Low → predicted Normal:")
print(top_low.head(10))
# summary beeswarm
shap.plots.beeswarm(sh_high2norm[1], max_display=15,
                    color_bar=False, show=True)

# waterfall plot of one representative row
idx = 0                 # any valid row index
shap.plots.waterfall(
    sh_high2norm[1][idx],   # note the extra [1] to pick the Normal slice
    max_display=15
)