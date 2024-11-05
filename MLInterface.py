#===- MLInterface.py - ACPO Python ML Interface --------------------------===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2021-2023. Huawei Technologies Co., Ltd. All rights reserved.
#
#===----------------------------------------------------------------------===//
module_imported = True


try:
  import os
  import sys
  import time
  import warnings
  warnings.simplefilter("ignore", UserWarning)

  from MLInference import *
except Exception as e:
  ACPO_LOG(str(e))
  module_imported = False


class Model:

  def __init__(self, num_features, num_outputs, model_inference, feature_pair,
               output_list, signature, model_dir, imported, infer, output_key,
               classes_dict, loadmodeltype):
    self.num_features = num_features
    self.num_outputs = num_outputs
    self.model_inference = model_inference
    self.feature_pair = feature_pair
    self.output_list = output_list
    self.signature = signature
    self.model_dir = model_dir
    self.imported = imported
    self.infer = infer
    self.output_key = output_key
    self.classes_dict = classes_dict
    self.loadmodeltype = loadmodeltype


def create_named_pipe(name):
  if os.path.exists(name):
    return 0
  os.mkfifo(name)
  return 1

def MLFSM(cmd_pipe, resp_pipe):
  """
    Now open the pipes, and make sure to keep the order consistent
    between here and the MLInterface. We need to create the response
    FIFO first to indicate to the LLVM-side that we got to this point.
    On LLVM side, the interface is waiting to see the creation of this
    response FIFO file. It will respond back by creating the command
    FIFO on which this side is waiting for. This completes the handshake.
    """
  responses = open(resp_pipe, "w")
  commands = open(cmd_pipe, "r")
  model_dict = {}
  inference_dict = {}
  active_model = ""
  active_inference = None
  output_dict = {}
  while True:
    line = commands.readline().rstrip()
    if not line:
      break
    else:
      ACPO_LOG("Received: %s at %s" % (line, time.time()))
      segments = line.split()
      CMD = segments[0]
      if (not module_imported):
        responses.write("ERROR in " + CMD + ": Module import error" + "\n")
        sys.exit(1)
      if CMD == "LoadModel":
        model_spec_file = segments[1]
        model_name = get_model_name(model_spec_file)
        if (model_dict.get(model_name) is not None):
          responses.write("Model loaded,already in dict," + model_name + "\n")
        else:
          try:
            load_result = load_model(model_spec_file)
            if (not load_result):
              responses.write(
                  "ERROR in LoadModel: Model could not be loaded\n")
              sys.exit(1)
            else:
              model_info_dict = load_result[0]
              num_features = len(model_info_dict.get(FEATURE_PAIR))
              num_outputs = len(model_info_dict.get(OUTPUT_LIST))
              new_model = Model(num_features, num_outputs,
                                model_info_dict.get(MODEL_INFERENCE),
                                model_info_dict.get(FEATURE_PAIR),
                                model_info_dict.get(OUTPUT_LIST),
                                model_info_dict.get(SIGNATURE),
                                model_info_dict.get(MODEL_DIRECTORY),
                                model_info_dict.get(IMPORTED),
                                model_info_dict.get(INFER),
                                model_info_dict.get(OUTPUT_KEY),
                                model_info_dict.get(CLASSES_DICT),
                                model_info_dict.get(LOADMODEL_TYPE))
              model_dict[model_name] = new_model
              inference_dict[model_name] = create_MLInference(
                  new_model.model_inference, new_model.model_dir,
                  new_model.infer, new_model.output_key,
                  new_model.classes_dict,
                  list(map(lambda o: o[0], new_model.output_list)), new_model.loadmodeltype)
              responses.write("Model loaded," + load_result[1] + "\n")

          except:
            responses.write("ERROR in LoadModel: An exception occurred\n")
            sys.exit(1)
        responses.flush()
      elif CMD == "InitializeFeatures":
        active_model = segments[1]
        active_inference = inference_dict.get(active_model)
        indices = segments[2::2]
        values = segments[3::2]
        features = []
        valid_index = True
        for i in range(len(indices)):
          index = int(indices[i])
          if (index < 0 or index >= model_dict.get(active_model).num_features):
            valid_index = False
            break
          feature = (index, values[i])
          features.append(feature)
        if (valid_index):
          try:
            active_inference.initialize_features(features)
            responses.write("Features initialized\n")

          except:
            responses.write(
                "ERROR in InitializeFeatures: An exception occurred\n")
            sys.exit(1)
        else:
          responses.write("ERROR in InitializeFeatures: Invalid index\n")
          sys.exit(1)
        responses.flush()
      elif CMD == "SetCustomFeature":
        index = int(segments[1])
        value = segments[2]
        if (index >= 0 and index < model_dict.get(active_model).num_features):
          try:
            active_inference.set_feature(index, value)
            responses.write("Feature set\n")
          except:
            responses.write(
                "ERROR in SetCustomFeature: An exception occurred\n")
            sys.exit(1)
        else:
          responses.write("ERROR in SetCustomFeature: Invalid index\n")
          sys.exit(1)
        responses.flush()
      elif CMD == "SetCustomFeatures":
        indices = segments[1::2]
        values = segments[2::2]
        features = []
        valid_index = True
        for i in range(len(indices)):
          index = int(indices[i])
          if (index < 0 or index >= model_dict.get(active_model).num_features):
            valid_index = False
            break
          feature = (index, values[i])
          features.append(feature)
        if (valid_index):
          try:
            active_inference.set_features(features)
            responses.write("Features set\n")
          except:
            responses.write(
                "ERROR in SetCustomFeatures: An exception occurred\n")
            sys.exit(1)
        else:
          responses.write("ERROR in SetCustomFeatures: Invalid index\n")
          sys.exit(1)
        responses.flush()
      elif CMD == "RunModel":
        output_dict = active_inference.runInfer()
        if (output_dict):
          responses.write("Completed\n")
        else:
          responses.write("ERROR in RunModel: Failed to run model " +
                          active_model + "\n")
        responses.flush()
      elif CMD == "GetModelOutput":
        output_name = segments[1]
        responses.write(output_name + ",int64_t," +
                        str(output_dict.get(output_name)) + "\n")
        responses.flush()
      elif CMD == "GetStatus":
        responses.write("Active\n")
        responses.flush()
      elif CMD == "ReleaseModel":
        model_name = segments[1]
        model_dict.pop(model_name, None)
        responses.write(model_name + ",model released\n")
        responses.flush()
      elif CMD == "CloseMLInterface":
        responses.write("Closing\n")
        responses.flush()
        break


# Main program is here.
command_pipe_name = sys.argv[1]
response_pipe_name = sys.argv[2]
ACPO_LOG("Starting ML Interface in Python\n")
if create_named_pipe(response_pipe_name) == 1:
  if create_named_pipe(command_pipe_name) == 1:
    MLFSM(command_pipe_name, response_pipe_name)
ACPO_LOG("Terminating ML interface\n")
os.remove(command_pipe_name)
os.remove(response_pipe_name)
