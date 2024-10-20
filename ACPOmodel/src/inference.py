'''
This script implements an standalone inference over the provided dataset
to evaluate the performance of a pre-trained model 
'''

# ------------------------- Import Required Libraries -------------------------
from __future__ import print_function

import csv
import json
import os
import pickle as pk
from collections import OrderedDict

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from sklearn.model_selection import *
from sklearn.preprocessing import *

from csv_process import *
from settings import *
from utils import *

# -------------------- Reading test data and preprocessing --------------------
args = sync_config(args.user_config)
os.chdir(args.root_path)

# set the path for work_dir and log_dir
data_dir = args.data["path"]
log_dir = os.path.join(args.log_path, args.model["name"])

work_dir  = args.work_path
model_dir = args.model["path"]
sc_file   = args.model["sc_file"]
cat_file  = args.model["cat_file"]

test_path  = args.test["path"]
test_file = args.test["file_name"]

class_key = args.class_key


# Inputs model
# pre_trained_model = os.path.abspath("../log/standalone/lu.pb")
# sc  = pk.load(open(os.path.abspath("../log/acpo-v3-model/sc.pkl"),'rb'))
# cat = pk.load(open(os.path.abspath("../data/lu-loocv/cats.pkl"),'rb'))
pre_trained_model = os.path.abspath(os.path.join(log_dir, model_dir))
sc_path           = os.path.abspath(os.path.join(log_dir,   sc_file))
cat_path          = os.path.abspath(os.path.join(data_dir, cat_file))
sc                = pk.load(open(sc_path,'rb'))
cats              = pk.load(open(cat_path,'rb'))


# generating inverse dictionary of cat: {int: keys}
classes_keys = dict()
classes_keys = { val: key for val, key in enumerate(cats) }
classes_keys = pd.DataFrame([classes_keys]).T

# For generating classes for those .csv files that doesn't, e.g., data-76.csv
# test_data = os.path.abspath("../data/lu/v2/15p/data-76.csv")
raw_test_data = os.path.join(test_path, test_file + ".csv")
dff           = pd.read_csv(raw_test_data, sep = ",")


# Computing all possible classes using combination of the features
# even some classes do not exist due to the lack of data
dff, _ = feature_to_class(dff, **args.feature_to_class)
print(f"Test-set: {dff.shape}")


# test_data = os.path.abspath("../work/data-76-classes.csv")
test_data = os.path.join(work_dir, "test/", test_file + "_processed" + ".csv")
dff.to_csv(test_data, index=False, sep=',')


# Load the pre_trained model
imported = tf.saved_model.load(pre_trained_model)
infer = imported.signatures["serving_default"]


# Import the test dataset
ds = pd.read_csv(test_data, sep=",")
print("\n--- Pruning Uncorrelated Features ---")
uncorr_ft = args.uncorrelated_features
ds.drop(uncorr_ft, axis=1, inplace=True)
features_start  = args.index["x_col_start"]
features_end    = args.index["x_col_end"]
features_length = args.index["y_col"]


# Output
filename = os.path.join(work_dir, "incorrect.csv") 


# Adding header to the csv file
# header = ['LoopIR','PredictionType', 'PredictionCount', 'ActualType', 'ActualCount']
header = args.header
with open(filename, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(header)

# Running inference pass over all rows
k = min(5, len(cats))
top_n = [0] * k

total_samples, _ = ds.shape
batch_size = total_samples        # when the total_samples is huge, use smaller batch_size
for i in range(0, total_samples, batch_size):

    samples = batch_size
    if (batch_size < total_samples) and (i + batch_size > total_samples):
        samples = total_samples % batch_size

    targets    = ds.iloc[i:i+samples, -1].to_numpy()
    ft         = ds.iloc[i:i+samples, features_start:features_end]
    ft         = sc.transform(np.array(ft).reshape(samples, -1))
    inputs     = np.array(ft, dtype=np.float32)
    inputs     = inputs.reshape(samples, features_length)
    outputs    = infer(tf.constant(inputs))
    outputs    = outputs['output_0'].numpy()

    k_labels = topK_labels(outputs, k)

    top_n = topk_metric(ds = ds.iloc[i:i+samples, :],
                        y = targets,
                        y_hat = outputs,
                        class_keys = classes_keys,
                        top_n = top_n,
                        filename = filename,
                        k = k,
                        id = args.id,
                        option=2
                        )

# calculating metrics for evaluating the model
top_accuracy = [100 * top_n[i]/total_samples for i in range(k)]

for i in range(k):
    print(f"Top-{i+1} Predictions: {top_accuracy[i]}")

# Writing to the end of the csv file
with open(filename, 'a+') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow('')
    for i in range(k):
        csvwriter.writerow([top_accuracy[i]])

