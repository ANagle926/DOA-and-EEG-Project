from collections import Counter
from hashlib import md5
import numpy as np
import pandas as pd
import shap
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.over_sampling import SMOTE
from joblib import load, dump
from matplotlib import pyplot as plt
from scipy import sparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import VotingClassifier, StackingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, RidgeClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GroupShuffleSplit, train_test_split, StratifiedGroupKFold, StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from interpret.glassbox import ExplainableBoostingClassifier

def process_data():
    x_test = load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/x_test_preop.joblib")
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

    df_train = pd.DataFrame(x_train, columns=all_cols)
    df_test  = pd.DataFrame(x_test,  columns=all_cols)


    #  ── LOW-MAC helper numeric & binary flags ------------------------------
    df_train["mac_fraction_low"] = (df_train["paO2"] < 0.8).astype(int)  # <- use real MAC var if you have it
    df_test ["mac_fraction_low"] = (df_test ["paO2"] < 0.8).astype(int)

    df_train["bmi_under_18"] = (df_train["bmi"] < 18).astype(int)
    df_test ["bmi_under_18"] = (df_test ["bmi"] < 18).astype(int)

    df_train["ph_acidotic"] = (df_train["ph"] < 7.30).astype(int)
    df_test ["ph_acidotic"] = (df_test ["ph"] < 7.30).astype(int)

    #  old flags you already had
    df_train["bmi_obese"]       = (df_train["bmi"] > 35).astype(int)
    df_test ["bmi_obese"]       = (df_test ["bmi"] > 35).astype(int)

    df_train["creatinine_high"] = (df_train["creatinine"] > 1.7).astype(int)
    df_test ["creatinine_high"] = (df_test ["creatinine"] > 1.7).astype(int)

    df_train["ph_alkalotic"]    = (df_train["ph"] > 7.5).astype(int)
    df_test ["ph_alkalotic"]    = (df_test ["ph"] > 7.5).astype(int)

    df_train["htn_creat_flag"]  = ((df_train["htn"] == 1) &
                                   (df_train["creatinine"] > 1.5)).astype(int)
    df_test ["htn_creat_flag"]  = ((df_test ["htn"] == 1) &
                                   (df_test ["creatinine"] > 1.5)).astype(int)


    flag_cols = [
        "bmi_obese", "creatinine_high", "ph_alkalotic",
        "htn_creat_flag",
        "mac_fraction_low", "ph_acidotic" #"bmi_under_18",
    ]

    numeric_feats = ["creatinine", "gpt", #removed pH
                     "paO2", "paCO2"] + flag_cols

    categorical_feats = ["dm", "anemia"] #removed htn

    # Build preprocessor
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_feats),          # unchanged
        ("cat", OneHotEncoder(handle_unknown='ignore'),
         categorical_feats)
    ])


    X_train_proc = preprocessor.fit_transform(df_train)
    X_test_proc  = preprocessor.transform(df_test)
    feat_names = preprocessor.get_feature_names_out()

    if sparse.issparse(X_train_proc):
        X_train_proc = X_train_proc.toarray()
        X_test_proc  = X_test_proc.toarray()

    train_idx, _ = train_test_split(
        np.arange(len(y_train)),
        train_size=100_000,
        stratify=y_train,
        random_state=42
    )
    X_small = X_train_proc[train_idx]
    y_small = y_train[train_idx]
    df_small = df_train.iloc[train_idx]

    """mask_high_sub = ((y_small == 2) &
                     (df_small["no_major_resection"] == 1) &
                     ((df_small["gender"] == 1) | (df_small["op_Hepatic"] == 1)))

    mask_low_sub  = ((y_small == 0) & (df_small["mac_fraction_low"] == 1))

    sampling_dict = {
        2: max(mask_high_sub.sum() * 10, Counter(y_small)[2]),
        0: max(mask_low_sub.sum()  * 10,  Counter(y_small)[0]),
        1: Counter(y_small)[1]
    }"""
    mask_high_sub = (
        (y_small == 2)
    )
    mask_low_sub = (
        (y_small == 0)
    )

    sampling_dict = {
        2: max(mask_high_sub.sum() * 20, Counter(y_small)[2]),
        0: max(mask_low_sub.sum()  * 20, Counter(y_small)[0]),
        1: Counter(y_small)[1]
    }

    smote = SMOTE(random_state=42, sampling_strategy=sampling_dict)
    X_res, y_res = smote.fit_resample(X_small, y_small)
    print("After targeted SMOTE:", Counter(y_res))

    return X_res, X_test_proc, y_res, y_test, feat_names, df_test

def create_model():

    penalty_cols = {"num__gender": 0.000005,  # ← 50× penalty versus default 1.0
                    "num__bmi":    0.0001,
                    "num__ph":     0.001}

    feature_weights = np.ones(len(feat_names))
    for col, w in penalty_cols.items():
        idx = np.where(feat_names == col)[0]
        if idx.size:
            feature_weights[idx[0]] = w


    TREE_FRAC = 0.8  # 0.80

    dt = DecisionTreeClassifier(max_depth=None,
                                class_weight="balanced",
                                max_features=TREE_FRAC,
                                random_state=42)


    et = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=None,
        max_features=TREE_FRAC,          # each tree sees half the columns
        class_weight='balanced',
        random_state=42)

    log_cv = LogisticRegressionCV(
        Cs=[0.01,0.1,1.0],
        penalty='l1',
        solver='liblinear',
        cv=5,
        max_iter=8000)

    bb = BalancedBaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=None, class_weight="balanced", max_features=TREE_FRAC),
        n_estimators=500,
        max_samples=1.0,
        bootstrap=False,            # avoid duplicates
        bootstrap_features=True,    # sub-sample features per estimator
        max_features=TREE_FRAC,
        random_state=42,
        n_jobs=1,
    )

    xgb = XGBClassifier(
        n_estimators=600,
        max_depth=6,
        min_child_weight=10,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        colsample_bynode=TREE_FRAC,   # weight vector honoured only if < 1.0
        feature_weights=feature_weights,
        eval_metric="mlogloss",
        random_state=42,
    )

    final_logi = LogisticRegression(
        penalty="l1",
        C=0.05,                   # stronger shrinkage than v1
        class_weight={0:15, 1:1, 2:10},
        solver="liblinear",
        max_iter=10_000,
    )

    stack = StackingClassifier(
        estimators=[
            ("xgb", xgb),
            ("dt", dt),
            ("LDA", LinearDiscriminantAnalysis()),
            #("CalibratedSVC", CalibratedClassifierCV(LinearSVC(max_iter=5000), cv=3)), #0.6890    and 0.6297    0.6176    0.6236
            #("ridge", RidgeClassifier()),
            ("mlp", MLPClassifier(random_state=42, max_iter=300))],
            #("bb", bb)],
        #final_estimator=final_logi,
        final_estimator=CalibratedClassifierCV(LinearSVC(max_iter=5000), cv=3),
        stack_method="predict_proba",
        cv=5,
        n_jobs=-1,
    )
    return stack

def train_stacked_model(x_train, x_test, y_train, y_test):

    stack = create_model()

    print("Fitting stack …")
    stack.fit(x_train, y_train)

    #stack= load("model_v7.joblib")

    print("Predicting probabilities …")
    y_proba = stack.predict_proba(x_test)

    TAU_HIGH = 0.53   #0.5 too low
    TAU_LOW  = 0.6   # 0.75 too low

    pred = np.full_like(y_test, 1)          # default Normal
    high = y_proba[:, 2] >= TAU_HIGH
    low  = (y_proba[:, 0] >= TAU_LOW) & (~high)
    pred[high] = 2
    pred[low]  = 0

    print("\nClassification report (test set):")
    print(classification_report(y_test, pred, digits=4))

    cm = confusion_matrix(y_test, pred, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay(cm, display_labels=["Low", "Normal", "High"]).plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Confusion Matrix – Test Set")
    plt.tight_layout(); plt.show()

    return pred, stack

def apply_thresholds(proba: np.ndarray, tau_low: float = 0.40, tau_high: float = 0.10):
    """Convert 3‑class probability matrix → integer labels.
    Priority: High first, then Low, else Normal.
    """
    pred = np.full(proba.shape[0], 1, dtype=int)  # default Normal
    high_mask = proba[:, 2] >= tau_high
    low_mask  = (proba[:, 0] >= tau_low) & (~high_mask)
    pred[high_mask] = 2
    pred[low_mask]  = 0
    return pred

def top_features(feat_names, sh_values, cls=1, k=15):

    vals = sh_values.values
    if vals.ndim == 3:          # C × n × p
        vals = vals[cls]        # take the class you care about
    imp   = np.abs(vals).mean(axis=0)
    order = np.argsort(imp)[::-1][:k]
    return pd.DataFrame({
        "feature": np.array(feat_names)[order],
        "mean_abs_shap": imp[order]
    })

def high_error_probe(X_proc, y_test, y_pred, feat_names):
#switched them
    # ── flag rows where true=High (2) but predicted=Normal (1) ──────────
    mask_high = y_test == 1.0
    y_err     = (y_pred[mask_high] == 2).astype(int)  # 1 = error
    X_high    = X_proc[mask_high]

    row_hash  = np.array([md5(r.tobytes()).hexdigest() for r in X_high])

    gss = GroupShuffleSplit(test_size=0.3, n_splits=1, random_state=42)
    train_idx, val_idx = next(gss.split(X_high, y_err, groups=row_hash))
    X_tr, X_va = X_high[train_idx], X_high[val_idx]
    y_tr, y_va = y_err[train_idx], y_err[val_idx]
    perfect_cols = [c for c in feat_names
                    if (X_tr[:, feat_names == c] == y_tr.reshape(-1, 1)).all()]
    print("Exact-copy leaks ➜", perfect_cols)

    corr = (pd.DataFrame(X_tr, columns=feat_names)
            .corrwith(pd.Series(y_tr, name="err"))
            .abs()
            .sort_values(ascending=False))
    print(corr.head(10))

    logi = LogisticRegression(max_iter=5000, penalty='l1', solver='liblinear')
    logi.fit(X_tr, y_tr)
    print(classification_report(y_va, logi.predict(X_va)))

    stump = DecisionTreeClassifier(max_depth=1).fit(X_tr, y_tr)
    print("stump acc:", stump.score(X_va, y_va))

def low_error_probe(X_proc, y_test, y_pred, feat_names, test_size=0.30, random_state=42):
#switched them
    mask_low  = y_test == 1.0
    y_err_low = (y_pred[mask_low] == 0).astype(int)
    X_low     = X_proc[mask_low]

    # --- 2. try group-aware split first ----------------------------
    row_hash  = np.array([md5(r.tobytes()).hexdigest() for r in X_low])
    gss = GroupShuffleSplit(test_size=test_size, n_splits=100,
                            random_state=random_state)

    for tr, va in gss.split(X_low, y_err_low, groups=row_hash):
        if len(np.unique(y_err_low[tr])) == 2 and len(np.unique(y_err_low[va])) == 2:
            tr_idx, va_idx = tr, va
            break
    else:
        # fallback: plain stratified split (allows duplicate rows)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size,
                                     random_state=random_state)
        tr_idx, va_idx = next(sss.split(X_low, y_err_low))

    X_tr, X_va = X_low[tr_idx], X_low[va_idx]
    y_tr, y_va = y_err_low[tr_idx], y_err_low[va_idx]

    ## --- 3. drop both gender dummies to avoid guaranteed leaks -----
    gender_cols = [i for i, n in enumerate(feat_names)
                   if n.startswith("cat__gender_")]
    keep_mask   = ~np.isin(np.arange(X_tr.shape[1]), gender_cols)

    X_tr, X_va         = X_tr[:, keep_mask], X_va[:, keep_mask]
    feat_keep          = feat_names[keep_mask]

    # --- 4. quick diagnostic models -------------------------------
    logi  = LogisticRegression(max_iter=5000, penalty="l1", solver="liblinear")
    stump = DecisionTreeClassifier(max_depth=1)

    logi.fit(X_tr, y_tr)
    stump.fit(X_tr, y_tr)

    print("Train/Val class balance:",
          Counter(y_tr), Counter(y_va), "\n")
    print(classification_report(y_va, logi.predict(X_va)))
    print("1-node stump accuracy:", stump.score(X_va, y_va))

    df_low = pd.DataFrame(X_test_proc[y_test == 0], columns=feat_names)
    df_low["err"] = (y_pred[y_test == 0] == 1).astype(int)
    #print(pd.crosstab(df_low["num__gender"], df_low["err"],
    #                  rownames=["gender"], colnames=["Low→Normal error?"]))

# --- 5. top correlations --------------------------------------
    df_tr  = pd.DataFrame(X_tr, columns=feat_keep)
    corr   = df_tr.corrwith(pd.Series(y_tr, name="err")).abs()
    print("\nTop features driving Low→Normal errors:")
    print(corr.sort_values(ascending=False).head(10))


X_res, X_test_proc, y_res, y_test, feat_names, df_test =process_data()

#dump(X_res, "x_res_v4.joblib")
#dump(X_test_proc, "x_test_proc_v4.joblib")
#dump(y_res, "y_res_v4.joblib")
#dump(y_test, "y_test_v4.joblib")
#dump(feat_names, "feat_names_v4.joblib")
#dump(df_test, "df_test_v4.joblib")
#print("finished saving data")"""

#X_test_proc= load("x_test_proc_v4.joblib")
#y_test = load("y_test_v4.joblib")
#X_res= load("x_res_v4.joblib")
#y_res= load("y_res_v4.joblib")
#feat_names= load("feat_names_v4.joblib")

y_pred, stack = train_stacked_model(X_res, X_test_proc, y_res, y_test)
dump(stack, "model_v7.joblib")
#dump(y_pred, "y_pred_v5.joblib")


high_error_probe(X_test_proc, y_test, y_pred, feat_names)
low_error_probe(X_test_proc, y_test, y_pred, feat_names)

high = (y_test == 2) & (y_pred == 1)  # High → Normal
low = (y_test == 0) & (y_pred == 1)  # Low  → Normal
print(f"High→Norm rows: {high.sum()}")
print(f"Low →Norm rows:  {low.sum()}")

xgb_fitted = stack.named_estimators_['xgb']
explainer = shap.TreeExplainer(xgb_fitted)   # xgb is the base learner object

X_high2norm = X_test_proc[high]     # high is the boolean mask you returned
X_low2norm  = X_test_proc[low]

rows = min(2000, X_high2norm.shape[0])
sh_high2norm = explainer(X_high2norm[:rows], check_additivity=False)
rows = min(2000, X_low2norm.shape[0])
sh_low2norm  = explainer(X_low2norm[:rows],  check_additivity=False)

top_high = top_features(feat_names, sh_high2norm, k=20)
top_low  = top_features(feat_names, sh_low2norm, k=20)

print("True High → predicted Normal:")
print(top_high.head(10))

print("\nTrue Low → predicted Normal:")
print(top_low.head(10))