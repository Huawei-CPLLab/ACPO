#===- DumpScalarInfo.py - Dump pickled data from sc.pkl -------------------===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2021-2023. Huawei Technologies Co., Ltd. All rights reserved.
#
#===----------------------------------------------------------------------===//
import numpy as np
import os
import pandas as pd
import pickle as pk
import tensorflow as tf
from sklearn.feature_selection import *

sc = pk.load(open("sc.pkl", "rb"))
print("Mean:")
for i in sc.mean_: print(i)
print("---------------------------------------")
print("Scale:")
for i in sc.scale_: print(i)
