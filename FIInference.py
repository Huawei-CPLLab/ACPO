#===- FIInference.py - ACPO Function Inlining Inference   ---------------===//
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
import torch
# import tensorflow as tf


fi_path = os.path.join(os.path.dirname(__file__), './models/pfi.pb')
fi_path = "/Users/lvyinrun/workspace/openEuler/ACPO_zm/ACPOmodel/src/log/inline_modelV0.0/"
frozen_model = fi_path
model_dir = fi_path
class FIInference(MLInference):

  def prepare_features(self):
    df = pd.DataFrame(self.features)
    pkl_file = os.path.join(os.path.dirname(__file__), model_dir, "sc.pkl")
    sc = pk.load(open(pkl_file, "rb"))
    print(df[0:56])
    df = sc.transform(df[0:56].transpose())
    input = np.array(df, dtype=np.float32)
    # size not right
    # input = input.reshape(1, len(self.features))
    input = input.reshape(1, 56)
    # return torch.utils.data.DataLoader(input)
    return input

  def should_inline(self, output):
    classes = [0, 1]
    max_class = 0
    max_val = output[0]
    for i, val in enumerate(output):
      if val > max_val:
        max_val = val
        max_class = i
    return classes[max_class]

  def inference(self):
    """
        Run an inference pass with an already loaded model and having features ready.
        This is for function inlining only for other inferences please see inference().
        """
    ACPO_LOG("ACPO Model successfully loaded for FI.")

    input = self.prepare_features()

    print(input)

    with torch.no_grad():
      output = self.infer(torch.from_numpy(input))
      print(output.tolist()[0][0])

    # output = self.infer(tf.constant(input))
    # output = output.get(self.output_key)
    output = output.tolist()[0][0];
    if (output is None):
      return {}
    output = output.numpy()

    output_dict = {}
    for i in range(len(self.output_names)):
      output_dict[self.output_names[i]] = self.should_inline(output[i])
      ACPO_LOG("Prediction is FI-ShouldInline=" +
            str(output_dict.get("FI-ShouldInline")) + "\n")
    return output_dict
