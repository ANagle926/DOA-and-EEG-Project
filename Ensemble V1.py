import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from joblib import load, dump
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import LinearSVC


x_test = load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/x_test_preop.joblib")
y_test = load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/y_test_preop.joblib")
x_train= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/x_train_preop.joblib")

static_cols = ["gender", "bmi", "htn", "dm", "anemia"]
surg_cols   = [
    "op_Biliary/Pancreas", "op_Breast", "op_Colorectal", "op_Hepatic",
    "op_Major resection", "op_Minor resection", "op_Others", "op_Stomach",
    "op_Thyroid", "op_Transplantation", "op_Vascular"
]

dynamic_cols = ["mean_sevo","std_sevo"]
all_cols = static_cols + surg_cols + dynamic_cols

numeric_feats   = ["bmi"] + dynamic_cols
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
dump(X_test_proc, "/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/x_test_preop.joblib")

X_res= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/x_smote_preop.joblib")
y_res= load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/y_smote_preop.joblib")


ensemble = VotingClassifier(
    estimators=[
        ('lr',  LogisticRegression(class_weight='balanced')),
        ('rf',  RandomForestClassifier(class_weight='balanced', max_depth=6)),
        ('cb',  CatBoostClassifier(auto_class_weights='Balanced', verbose=False)),
        ('svc', CalibratedClassifierCV(LinearSVC(max_iter=5000), cv=3))
    ],
    voting='soft',
    weights=[1.2, 1, 1, 1.5],  # you can increase the weight of the model that does best on class 1
    n_jobs=-1
)

stack = StackingClassifier(
    estimators=[
        ('lr', LogisticRegression(class_weight='balanced')),
        ('rf', RandomForestClassifier(class_weight='balanced', max_depth=6)),
        ('cb', CatBoostClassifier(auto_class_weights='Balanced', verbose=False)),
        ('svc', CalibratedClassifierCV(LinearSVC(max_iter=5000), cv=3))
    ],
    final_estimator=LogisticRegression(class_weight='balanced'),
    cv=5,
    n_jobs=-1
)

ensemble.fit(X_res, y_res)
stack.fit(X_res, y_res)


y_proba_ensemble = ensemble.predict_proba(X_test_proc)
thresh1 = 0.35  # lower or higher to bias recall/precision
y_pred_ensemble = np.where(y_proba_ensemble[:,1] >= thresh1, 1, np.argmax(y_proba_ensemble[:, [0,2]], axis=1) * 2)


y_proba_stack = stack.predict_proba(X_test_proc)
thresh1 = 0.35  # lower or higher to bias recall/precision
y_pred_stack = np.where(y_proba_stack[:,1] >= thresh1, 1, np.argmax(y_proba_stack[:, [0,2]], axis=1) * 2)


print(classification_report(y_test, y_pred_ensemble, digits=4))
print(classification_report(y_test, y_pred_stack,    digits=4))