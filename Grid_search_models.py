from collections import Counter
import numpy as np
import pandas as pd
from IPython.core.display_functions import display
from catboost import CatBoostClassifier
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE
from joblib import dump, load
from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.ensemble            import HistGradientBoostingClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process     import GaussianProcessClassifier
from pytorch_tabnet.tab_model     import TabNetClassifier
from imblearn.ensemble            import EasyEnsembleClassifier

from VitalDBDataset import VitalDBDataset


"""vitaldb= VitalDBDataset(num_cases=4000, srate=128)

x_train= vitaldb.x_train
x_test= vitaldb.x_test
y_train= vitaldb.y_train
y_test= vitaldb.y_test
c_train= vitaldb.c_train
c_test= vitaldb.c_test

dump(x_train, "/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/x_train_preop_age.joblib")
dump(x_test, "/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/x_test_preop_age.joblib")
dump(y_train, "/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/y_train_preop_age.joblib")
dump(y_test, "/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/y_test_preop_age.joblib")
dump(c_train, "/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/c_train_preop_age.joblib")
dump(c_test, "/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/c_test_preop_age.joblib")"""

def process_data(x_train, x_test, y_train, y_test, smote):

    surg_cols   = [
        "op_Biliary/Pancreas", "op_Breast", "op_Colorectal", "op_Hepatic",
        "op_Major resection", "op_Minor resection", "op_Others", "op_Stomach",
        "op_Thyroid", "op_Transplantation", "op_Vascular"
    ]

    num_feats   = ["bmi", "creatinine", "gpt", "age", "hemoglobin"]  # scale
    cat_feats   = ["gender", "surgery_type", "physical_stat"]         # one-hot
    bin_feats   = ["hypertension", "diabetes"]                        # already 0/1


    all_cols = [ "gender", "bmi", "surgery_type", "hypertension", "diabetes", "hemoglobin", "creatinine", "gpt", "physical_stat", "age"  ]

    df_train = pd.DataFrame(x_train, columns=all_cols)
    df_test  = pd.DataFrame(x_test,  columns=all_cols)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num",  StandardScaler(), num_feats),
            ("cat",  OneHotEncoder(handle_unknown="ignore",sparse_output=False),  cat_feats),
            ("bin",  "passthrough",  bin_feats),
        ],
        remainder="drop",
    )


    x_train_proc = preprocessor.fit_transform(df_train)
    x_test_proc  = preprocessor.transform(df_test)

    print(f"Original training dataset shape: {Counter(y_train)}")
    X_res, y_res = smote.fit_resample(x_train_proc, y_train)
    print(f"Resampled training dataset shape: {Counter(y_res)}")

    return X_res, y_res, x_test_proc, y_test

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

def plot_confusion_matrix(y_true, y_pred, model_name, smote_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["low","normal","high"],
                yticklabels=["low","normal","high"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix for {model_name} with {smote_name}")
    plt.show()


x_train= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/x_train_preop_age.joblib")
x_test= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/x_test_preop_age.joblib")
y_train= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/y_train_preop_age.joblib")
y_test= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/y_test_preop_age.joblib")

models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Ridge Classifier': RidgeClassifier(),
    'MLP Classifier': MLPClassifier(random_state=42, max_iter=300),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Extra Trees': ExtraTreesClassifier(random_state=42),
    'Decision Trees': DecisionTreeClassifier(random_state=42),
    'SVC': SVC(random_state=42, probability=True),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    "XGB": XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42, use_label_encoder=False, val_metric="mlogloss"),
    "CalibratedSVC": CalibratedClassifierCV(LinearSVC(max_iter=5000), cv=3),
    "LightGBM": LGBMClassifier(n_estimators=200, learning_rate=0.1, max_depth=  4,random_state=42, verbose=-1),
    "CatBoost": CatBoostClassifier(iterations=200, learning_rate=0.1, depth=6, random_seed=42, verbose=False),
    "CBBalanced":CatBoostClassifier(iterations=200, auto_class_weights="Balanced",random_seed=42, verbose=False),
    "LGBBalanced": LGBMClassifier(n_estimators=200, class_weight="balanced", random_state=42, verbose=-1),
    "BalancedBagging": BalancedBaggingClassifier(estimator=DecisionTreeClassifier(max_depth=6),sampling_strategy="auto", n_estimators=10, max_samples=0.5, replacement=True, random_state=42,n_jobs=1),
    "RandomForestBalanced": RandomForestClassifier(n_estimators=200,max_depth=6, class_weight="balanced",random_state=42,n_jobs=-1),
    "LDA": LinearDiscriminantAnalysis(),
    "GNB": GaussianNB(),
    'EasyEnsemble': EasyEnsembleClassifier(n_estimators=50, random_state=42),
    'QDA': QuadraticDiscriminantAnalysis(reg_param=0.1),
    #'TabNet': TabNetClassifier(n_d=32, n_steps=5, gamma=1.5, seed=42, verbose=0, device_name='cpu'),
    'HistGradientBoosting': HistGradientBoostingClassifier(max_depth=None, learning_rate=0.1, l2_regularization=0.0, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=200, learning_rate=0.5, random_state=42),
    'GaussianProcess': GaussianProcessClassifier(random_state=42),
    "SGD" : SGDClassifier(loss="log_loss", penalty="elasticnet",class_weight="balanced", random_state=42),
}

smote_options = {
    #"Vanilla SMOTE": SMOTE(random_state=42),
    #"Borderline-1": BorderlineSMOTE(kind="borderline-1", random_state=42),
    #"Borderline-2": BorderlineSMOTE(kind="borderline-2", random_state=42),
    "SMOTE (all)": SMOTE(sampling_strategy="not majority", random_state=42),
    "SVM-SMOTE": SVMSMOTE(sampling_strategy="not majority", random_state=42),
    "SMOTE-Tomek": SMOTETomek(sampling_strategy="not majority"),
    "SMOTE-ENN": SMOTEENN(sampling_strategy="not majority"),
}


for smote_name, smote in smote_options.items():
    print(f"=== Using {smote_name} ===")
    results = []
    X_res, y_res, X_test_proc, y_test = process_data(x_train, x_test, y_train, y_test, smote)

    for model_name, model in tqdm(models.items(), desc="Training and Evaluating Models"):
        model.fit(X_res, y_res)
        y_pred = model.predict(X_test_proc)
        y_pred_proba = model.predict_proba(X_test_proc) if hasattr(model, "predict_proba") else None

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
        plot_confusion_matrix(y_test, y_pred, model_name, smote_name)

    # Display as a styled table
    results_df = pd.DataFrame(results)
    styled_results_df = results_df.style.set_table_styles(
        [{'selector': 'th', 'props': [('font-weight', 'bold'), ('background-color', '#f0f0f0')]},
         {'selector': 'td', 'props': [('padding', '5px')]}]
    ).set_properties(**{'text-align': 'center'}).set_caption("Model Evaluation Results with SMOTE")

    display(styled_results_df)


