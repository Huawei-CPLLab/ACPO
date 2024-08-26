#!/usr/bin/env python3

# -------------- Import required libraries & Synchronize settings --------------
import os

import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

os.sys.path.append("../../src")
from csv_process import *  # CSV data preprocessing
from settings import *  # Training settings, sync function

if args.root_path:
    os.chdir(args.root_path)

# --------------------------- Synchronize settings ----------------------------
args = sync_config()


# ---------------------------- Importing dataset ------------------------------
# read data csv into DataFrame
raw_data_path = os.path.join(args.data["path"], args.data["file_name"] + ".csv")
df = pd.read_csv(raw_data_path, sep=",")


# ---------------------------- Sanitizing dataset -----------------------------
remove_duplicates(df, **args.remove_duplicates)
prune_data(df, **args.prune_data)
remove_null(df)


# ------------------------- Feature Cleaning ----------------------------------
drop_feature(df, **args.drop_feature)
remove_constant_feature(df, **args.remove_constant_feature)


# ------------------------- Adding class to DataFrame -------------------------
df, cat = feature_to_class(df, **args.feature_to_class)
df = df.sample(frac=1).reset_index(drop=True)


# ---------------------------- Save Sanitized dataset -------------------------
# if args.data["save_file"]:
#    processed_data_path = os.path.join(args.data_path, args.processed_file + ".csv")
#    df.to_csv(processed_data_path, index=False, sep=",")


# ------------------------- Data Statistics -----------------------------------
print(df.head())
print(df.describe())


# ---------------------------- Exploring Features -----------------------------
print(f"for the dataset with rows, columns of {df.shape}")
print(features_relation(df, **args.features_relation))


# ----------------------- Reporting Features Importance -----------------------
# prepare dataset for training
start_idx = args.index["x_col_start"]
end_idx = args.index["x_col_end"]
X = df.iloc[:, start_idx:end_idx]
y = df["Classes"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=12, stratify=y
)

# train a model with RandomForest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

# predict labels
y_pred_test = rf.predict(X_test)
y_pred_train = rf.predict(X_train)

# reports
print("\n On a preliminary trained model:")
print(f"\t Test accuracy {100 * accuracy_score(y_test, y_pred_test):.2f}%")
print(f"\t Train accuracy {100 * accuracy_score(y_train, y_pred_train):.2f}%\n")


train_confusion_matrix = confusion_matrix(y_train, y_pred_train)
test_confusion_matrix = confusion_matrix(y_test, y_pred_test)
print(f"confusion matrix on training data: \n {train_confusion_matrix} \n")
print(f"confusion matrix on test data: \n {test_confusion_matrix} \n")
print("Horizontal axis shows predicted labels, vertical axis depicts true labels. \n")


correct_train = np.diag(train_confusion_matrix)
correct_test = np.diag(test_confusion_matrix)
incorrect_train = np.sum(train_confusion_matrix, axis=1) - np.diag(
    train_confusion_matrix
)
incorrect_test = np.sum(test_confusion_matrix, axis=1) - np.diag(test_confusion_matrix)

prediction_summary_data = np.array(
    [correct_train, correct_test, incorrect_train, incorrect_test]
).T
mux = pd.MultiIndex.from_product([["correct", "incorrect"], ["train", "test"]])
prediction_count = pd.DataFrame(prediction_summary_data, columns=mux)

print(f"Prediction summary: \n {prediction_count} \n")

correct_tr_percentage = correct_train / np.sum(train_confusion_matrix)
correct_te_percentage = correct_test / np.sum(test_confusion_matrix)
incorrect_tr_percentage = incorrect_train / np.sum(train_confusion_matrix)
incorrect_te_percentage = incorrect_test / np.sum(test_confusion_matrix)

prediction_percentage_data = np.array(
    [
        correct_tr_percentage,
        correct_te_percentage,
        incorrect_tr_percentage,
        incorrect_te_percentage,
    ]
).T
mux = pd.MultiIndex.from_product([["correct %", "incorrect %"], ["train", "test"]])
prediction_percentage = pd.DataFrame(prediction_percentage_data, columns=mux).round(
    decimals=3
)

print(f"Prediction distribution: \n {prediction_percentage} \n")


train_classification_report = classification_report(y_train, y_pred_train)
test_classification_report = classification_report(y_test, y_pred_test)
print(f"Classification report on train data: \n {train_classification_report} \n")
print(f"Classification report on test data: \n {test_classification_report} \n")

print(
    """
Definitions:
* 'Precision' is the number of correctly-identified members of a class divided by all the times the model predicted that class.
   In the case of Cat/Dog, the precision score would be the number of correctly-identified Cat divided by the total number of times the classifier predicted "Cat," rightly or wrongly.

* 'Recall' is the number of members of a class that the classifier identified correctly divided by the total number of members in that class.
   For Cat, this would be the number of actual Cats that the classifier correctly identified as such."

* 'F1 score' is a little less intuitive because it combines precision and recall into one metric.
   If precision and recall are both high, F1 will be high, too. If they are both low, F1 will be low. If one is high and the other low, F1 will be low.
   F1 is a quick way to tell whether the classifier is actually good at identifying members of a class, or if it is finding shortcuts (e.g., just identifying everything as a member of a large class).
"""
)


# Reporting Feature importance on method 1
feature_importance = rf.feature_importances_
sorted_idx = feature_importance.argsort()[::-1]
feature_importance = pd.DataFrame(
    zip(X.columns[sorted_idx], feature_importance[sorted_idx])
)
print(f"Method 1: feature importance and its value \n {feature_importance} \n")


# Reporting Feature importance on method 2
feature_importance = permutation_importance(rf, X_test, y_test).importances_mean
sorted_idx = feature_importance.argsort()[::-1]
feature_importance = pd.DataFrame(
    zip(X.columns[sorted_idx], feature_importance[sorted_idx])
)
print(f"Method 2: feature importance and its value \n {feature_importance} \n")


# Reporting Feature importance on method 3
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)
n_class = len(shap_values)
n_features = shap_values[0].shape[1]
df_shap = pd.DataFrame(
    columns=["class_" + str(i) for i in range(n_class)], index=X.columns
)

for i in range(n_class):
    df_shap["class_" + str(i)] = np.mean(np.abs(shap_values[i]), axis=0)

df_shap["mean|shap|"] = df_shap.mean(axis=1).to_frame()
df_shap = df_shap.sort_values("mean|shap|", ascending=False)
print(f"Method 3: feature importance and its value \n {df_shap}")
