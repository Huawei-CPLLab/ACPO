#===- LUInference.py - ACPO Loop Unroll Inference   ----------------------===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2021-2023. Huawei Technologies Co., Ltd. All rights reserved.
#
#===----------------------------------------------------------------------===//
from MLInference import MLInference, ACPO_LOG
import numpy as np
import os
import pandas as pd
import pickle as pk
import tensorflow as tf

# Must be a full path
lu_path = os.path.join(os.path.dirname(__file__), './models/plu.pb')
frozen_model = lu_path
model_dir = lu_path

# Optional (not being used right now): inference from a csv containing features
test_features = lu_path + '/test_features.csv'

LU_Enum_Dic = {0: 'full', 1: 'partial', 2: 'runtime', 3: 'unmodified'}


class LUInference(MLInference):

  def prepare_features(self):
    """
        Process the features and convert them into an input to be used for inference.
        """
    df = pd.DataFrame(self.features)
    #print("Feature arrived at Python side:\n", features)
    pkl_file = os.path.join(os.path.dirname(__file__), model_dir, "sc.pkl")
    sc = pk.load(open(pkl_file, "rb"))
    df = sc.transform(df.transpose())
    input = np.array(df, dtype=np.float32)
    input = input.reshape(1, len(self.features))
    return input

  def inference(self):
    """
        Run an inference pass with an already loaded model and having features ready
        """
    ACPO_LOG("ACPO Model successfully loaded.")
    input = self.prepare_features()

    output = self.infer(tf.constant(input))

    output = output.get(self.output_key)
    if (output is None):
      return {}
    output = output.numpy()
    output_class_index = np.argmax(output)
    # For now we assume that the order of outputs within the output class tuple is the same
    # as the order specified by the OutputList field in model.acpo.
    # A better way might be generating a dict of dicts or a list of dicts during load_model.
    # Then use index to get the output dict directly. The dict will look like this:
    # classes_dict = {0: {'LU-Count': 0, 'LU-Type': 3}, 1: {'LU-Count': 0, 'LU-Type': 2}, ...}
    output_dict = {}
    output_class = self.classes_dict.get(output_class_index)
    for i in range(len(self.output_names)):
      output_dict[self.output_names[i]] = output_class[i]
      ACPO_LOG("Prediction is UP.Count=" + str(output_dict.get("LU-Count")) +
            "\n")
    return output_dict
