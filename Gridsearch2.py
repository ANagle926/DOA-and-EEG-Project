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
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.ensemble            import HistGradientBoostingClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process     import GaussianProcessClassifier
from interpret.glassbox           import ExplainableBoostingClassifier
from pytorch_tabnet.tab_model     import TabNetClassifier
from imblearn.ensemble            import EasyEnsembleClassifier

from VitalDBDataset import VitalDBDataset


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
                xticklabels=["normal", "abnormal"],
                yticklabels=["normal", "abnormal"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix for {model_name}")
    plt.show()


x_train= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/x_train_preop.joblib")
x_test= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/x_test_preop.joblib")
y_train= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/y_train_preop.joblib")
y_test= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/y_test_preop.joblib")

models = {
    #'Logistic Regression': LogisticRegression(random_state=42),
    #'Ridge Classifier': RidgeClassifier(),
    #'MLP Classifier': MLPClassifier(random_state=42, max_iter=300),
    #'Random Forest': RandomForestClassifier(random_state=42),
    #'Extra Trees': ExtraTreesClassifier(random_state=42),
    #'Decision Trees': DecisionTreeClassifier(random_state=42),
    #'SVC': SVC(random_state=42, probability=True),
    #'K-Nearest Neighbors': KNeighborsClassifier(),
    #"XGB": XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42, use_label_encoder=True, val_metric="mlogloss"),
    #"CalibratedSVC": CalibratedClassifierCV(LinearSVC(max_iter=5000), cv=3),
    #"LightGBM": LGBMClassifier(n_estimators=200, learning_rate=0.1, max_depth=  4,random_state=42, verbose=-1),
    #"CatBoost": CatBoostClassifier(iterations=200, learning_rate=0.1, depth=6, random_seed=42, verbose=False),
    #"CBBalanced":CatBoostClassifier(iterations=200, auto_class_weights="Balanced",random_seed=42, verbose=False),
    #"LGBBalanced": LGBMClassifier(n_estimators=200, class_weight="balanced", random_state=42, verbose=-1),
    #"BalancedBagging": BalancedBaggingClassifier(estimator=DecisionTreeClassifier(max_depth=6),sampling_strategy="auto", n_estimators=10, max_samples=0.5, replacement=True, random_state=42,n_jobs=1),
    #"RandomForestBalanced": RandomForestClassifier(n_estimators=200,max_depth=6, class_weight="balanced",random_state=42,n_jobs=-1),
    #"LDA": LinearDiscriminantAnalysis(),
    #"GNB": GaussianNB(),
    #'EasyEnsemble': EasyEnsembleClassifier(n_estimators=50, random_state=42),
    #'QDA': QuadraticDiscriminantAnalysis(reg_param=0.1),
    #'TabNet': TabNetClassifier(n_d=32, n_steps=5, gamma=1.5, seed=42, verbose=0, device_name='cpu'),
    'HistGradientBoosting': HistGradientBoostingClassifier(max_depth=None, learning_rate=0.1, l2_regularization=0.0, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=200, learning_rate=0.5, random_state=42),
    'GaussianProcess': GaussianProcessClassifier(random_state=42),
    "SGD" : SGDClassifier(loss="log_loss", penalty="elasticnet",class_weight="balanced", random_state=42),
}

results = []

x_train_proc, y_train_proc, x_test_proc, y_test_proc, preprocessor= process_data(x_train, x_test, y_train, y_test)

feature_names = preprocessor.get_feature_names_out()

x_train_proc  = pd.DataFrame(x_train_proc, columns=feature_names)
x_test_proc = pd.DataFrame(x_test_proc, columns=feature_names)

for model_name, model in tqdm(models.items(), desc="Training and Evaluating Models"):
    model.fit(x_train_proc, y_train_proc)
    y_pred = model.predict(x_test_proc)
    y_pred_proba = model.predict_proba(x_test_proc) if hasattr(model, "predict_proba") else None
    plot_confusion_matrix(y_test_proc, y_pred, model_name)

# Display as a styled table
results_df = pd.DataFrame(results)
styled_results_df = results_df.style.set_table_styles(
    [{'selector': 'th', 'props': [('font-weight', 'bold'), ('background-color', '#f0f0f0')]},
     {'selector': 'td', 'props': [('padding', '5px')]}]
).set_properties(**{'text-align': 'center'}).set_caption("Model Evaluation Results with SMOTE")

display(styled_results_df)

