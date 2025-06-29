from collections import Counter
from imblearn.ensemble import BalancedRandomForestClassifier, BalancedBaggingClassifier
from imblearn.over_sampling import SMOTE
from joblib import load, dump
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

from VitalDBDataset import VitalDBDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, auc
from tqdm import tqdm
from IPython.display import display
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier



"""def visualize_performance():

    # We can hard-code visual themes and parameters
    sns.set_theme(style="ticks")
    plt.rcParams.update({
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.dpi': 300,
        'savefig.bbox': 'tight',
        'lines.linewidth': 2.5,
    })

    # Using the log_reg_model from above
    y_pred = log_reg_model.predict(X_test_scaled)
    y_prob = log_reg_model.predict_proba(X_test_scaled)[:, 1]

    # First plot ROC Curve with Bootstrapping and 95% CI
    bootstraps = 1000
    roc_curves = []

    for _ in range(bootstraps):
        X_test_resampled, y_test_resampled = resample(X_test_scaled, y_test, random_state=42)
        y_prob_resampled = log_reg_model.predict_proba(X_test_resampled)[:, 1]
        fpr_resampled, tpr_resampled, _ = roc_curve(y_test_resampled, y_prob_resampled)
        roc_curves.append((fpr_resampled, tpr_resampled))

    mean_fpr = np.linspace(0, 1, 100)
    tprs = [np.interp(mean_fpr, fpr, tpr) for fpr, tpr in roc_curves]
    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)
    tpr_upper = np.minimum(mean_tpr + 1.96 * std_tpr, 1)
    tpr_lower = mean_tpr - 1.96 * std_tpr

    plt.figure(figsize=(8, 8))
    plt.plot(mean_fpr, mean_tpr, color='#D32D41', label=f"Mean ROC Curve (AUC = {auc(mean_fpr, mean_tpr):.2f})")
    plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color='#D32D41', alpha=0.3, label="95% CI")
    plt.plot([0, 1], [0, 1], linestyle='--', color='grey', linewidth=1.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right", frameon=False)
    plt.grid(visible=True, which='both', linestyle=':', linewidth=0.5)
    plt.show()

    # Second, Precision-Recall Curve with Bootstrapping and 95% CI
    pr_curves = []
    for _ in range(bootstraps):
        X_test_resampled, y_test_resampled = resample(X_test_scaled, y_test, random_state=42)
        y_prob_resampled = log_reg_model.predict_proba(X_test_resampled)[:, 1]
        precision_resampled, recall_resampled, _ = precision_recall_curve(y_test_resampled, y_prob_resampled)
        pr_curves.append((recall_resampled, precision_resampled))

    mean_recall = np.linspace(0, 1, 100)
    precisions = [np.interp(mean_recall, recall[::-1], precision[::-1]) for recall, precision in pr_curves]
    mean_precision = np.mean(precisions, axis=0)
    std_precision = np.std(precisions, axis=0)
    precision_upper = np.minimum(mean_precision + 1.96 * std_precision, 1)
    precision_lower = mean_precision - 1.96 * std_precision

    plt.figure(figsize=(8, 8))
    plt.plot(mean_recall, mean_precision, color='#488A99', label="Mean Precision-Recall Curve")
    plt.fill_between(mean_recall, precision_lower, precision_upper, color='#488A99', alpha=0.3, label="95% CI")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left", frameon=False)
    plt.grid(visible=True, which='both', linestyle=':', linewidth=0.5)
    plt.show()

    # Third, Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, annot_kws={"size": 16})
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.xticks([0.5, 1.5], ["No Disease", "Disease"], fontsize=14)
    plt.yticks([0.5, 1.5], ["No Disease", "Disease"], fontsize=14)
    plt.show()

    # Fourth, Permutation Feature Importance with Custom Error Bars
    try:
        # Retrieve feature names
        encoded_feature_names = preprocessor.transformers_[1][1]['encoder'].get_feature_names_out(categorical_cols)
        feature_names = list(numerical_cols) + list(encoded_feature_names)
    except AttributeError:
        feature_names = x_train.columns

    # Calculate Permutation Importance
    perm_importance = permutation_importance(log_reg_model, X_test_scaled, y_test, n_repeats=30, random_state=42)

    # Check if feature names and importances match in length; if they don't, raise an informative error
    if len(feature_names) != len(perm_importance.importances_mean):
        raise ValueError(f"Mismatch in feature names ({len(feature_names)}) and importances ({len(perm_importance.importances_mean)}). "
                         "Check the preprocessor or feature extraction pipeline for consistency.")

    # Create df for feature importances
    perm_importances_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': perm_importance.importances_mean,
        'Std': perm_importance.importances_std
    }).sort_values(by="Importance", ascending=False)

    # Clean up NaN or inf values
    perm_importances_df['Importance'].replace([np.inf, -np.inf], np.nan, inplace=True)
    perm_importances_df['Std'].replace([np.inf, -np.inf], np.nan, inplace=True)
    perm_importances_df.fillna(0, inplace=True)

    # Plot
    plt.figure(figsize=(10, 8))
    bars = plt.barh(perm_importances_df['Feature'], perm_importances_df['Importance'], color='skyblue', xerr=perm_importances_df['Std'])
    plt.xlabel("Mean Decrease in Accuracy")
    plt.ylabel("Feature")
    plt.title("Permutation Feature Importance")
    plt.gca().invert_yaxis()  # Invert y-axis

    # Error bars
    for i, bar in enumerate(bars):
        plt.errorbar(
            bar.get_width(), i, xerr=perm_importances_df['Std'].iloc[i],
            fmt='o', color='grey', capsize=5
        )

    plt.show()"""
def visualize_data(x_train, y_train):
    # Now no 'age' column:
    static_cols = ["gender", "bmi", "htn", "dm", "anemia"]

    surg_cols = [
        "op_Biliary/Pancreas", "op_Breast", "op_Colorectal", "op_Hepatic",
        "op_Major resection", "op_Minor resection", "op_Others", "op_Stomach",
        "op_Thyroid", "op_Transplantation", "op_Vascular"
    ]

    all_cols = static_cols + surg_cols

    # Build DataFrame
    df_train = pd.DataFrame(x_train, columns=all_cols)
    df_train["MAC_class"] = y_train  # 0=low,1=normal,2=high

    # 1) Class balance bar chart
    df_train["MAC_class"].value_counts().sort_index().plot(
        kind="bar", edgecolor="black"
    )
    plt.xlabel("MAC class")
    plt.ylabel("Count")
    plt.title("Class distribution (0=low,1=normal,2=high)")
    plt.show()

    # 2) Histogram of BMI
    plt.hist(df_train["bmi"], bins=30, edgecolor="black")
    plt.xlabel("BMI")
    plt.title("BMI distribution")
    plt.show()

    # 3) Scatter of BMI vs. class (with jitter on y for visibility)
    y_jitter = df_train["MAC_class"] + (np.random.rand(len(df_train)) - 0.5)*0.1
    plt.scatter(df_train["bmi"], y_jitter, alpha=0.3)
    plt.xlabel("BMI")
    plt.ylabel("MAC class")
    plt.title("BMI vs. MAC class")
    plt.yticks([0, 1, 2])
    plt.show()

    # 4) Boxplots of BMI by a few categorical features
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    cats = ["gender", "htn", "dm", "anemia", surg_cols[0], surg_cols[1]]

    for ax, col in zip(axes.flatten(), cats):
        df_train.boxplot(column="bmi", by=col, ax=ax)
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel("BMI")
    plt.suptitle("")   # remove the automatic suptitle
    plt.tight_layout()
    plt.show()

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

dataset= VitalDBDataset(num_cases=750)

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
dump(c_test, "/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/c_test_preop.joblib")


"""x_train= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/x_train_preop.joblib")
x_test = load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/x_test_preop.joblib")
y_train= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/y_train_preop.joblib")
y_test = load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/y_test_preop.joblib")"""

print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)

visualize_data(x_train, y_train)

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
    ("cat", OneHotEncoder(),     categorical_feats),
])

# Convert x_train/x_test into DataFrames
df_train = pd.DataFrame(x_train, columns=all_cols)
df_test  = pd.DataFrame(x_test,  columns=all_cols)

X_train_proc = preprocessor.fit_transform(df_train)
X_test_proc  = preprocessor.transform(df_test)

# SMOTE which over-samples the training set
print(f"Original training dataset shape: {Counter(y_train)}")
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train_proc, y_train)
print(f"Resampled training dataset shape: {Counter(y_res)}")

dump(X_res, "/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/x_smote_preop.joblib")
dump(y_res, "/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/y_smote_preop.joblib")

"""X_res= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/x_smote_preop.joblib")
y_res= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/y_smote_preop.joblib")"""

models = {

    'Logistic Regression': LogisticRegression(random_state=42),
    'Ridge Classifier': RidgeClassifier(),
    'MLP Classifier': MLPClassifier(random_state=42, max_iter=300),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Extra Trees': ExtraTreesClassifier(random_state=42),
    'Decision Trees': DecisionTreeClassifier(random_state=42),
    'SVC': SVC(random_state=42, probability=True),
    'K-Nearest Neighbors': KNeighborsClassifier()
}

# Compare model metrics with a great table
results = []

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
    plot_confusion_matrix(y_test, y_pred, model_name)

# Display as a styled table
results_df = pd.DataFrame(results)
styled_results_df = results_df.style.set_table_styles(
    [{'selector': 'th', 'props': [('font-weight', 'bold'), ('background-color', '#f0f0f0')]},
     {'selector': 'td', 'props': [('padding', '5px')]}]
).set_properties(**{'text-align': 'center'}).set_caption("Model Evaluation Results with SMOTE")

display(styled_results_df)
