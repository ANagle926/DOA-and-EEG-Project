import json

import joblib
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

from VitalDBDataset import VitalDBDataset

dataset= VitalDBDataset(num_cases=750)

x_train= dataset.x_train
x_test= dataset.x_test
y_train= dataset.y_train
y_test= dataset.y_test

dump(x_train, "/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/x_train_preop.joblib")
dump(x_test, "/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/x_test_preop.joblib")
dump(y_train, "/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/y_train_preop.joblib")
dump(y_test, "/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/y_test_preop.joblib")

# Define full feature names
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

joblib.dump(preprocessor, "preprocessor.joblib")
metadata = {
    "static_cols": ["gender", "bmi", "htn", "dm", "anemia"],
    "surg_cols":   [
        "op_Biliary/Pancreas", "op_Breast", "op_Colorectal", "op_Hepatic",
        "op_Major resection", "op_Minor resection", "op_Others", "op_Stomach",
        "op_Thyroid", "op_Transplantation", "op_Vascular"
    ],
    "dynamic_cols": ["mean_sevo", "std_sevo"]
}
with open("feature_metadata.json", "w") as f:
    json.dump(metadata, f)
