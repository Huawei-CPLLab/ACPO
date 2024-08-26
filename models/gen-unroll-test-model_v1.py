#===- gen-unroll-test-model_v1.py - Generate Loop Unroll Model ------------===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2021-2023. Huawei Technologies Co., Ltd. All rights reserved.
#
#===----------------------------------------------------------------------===//

"""Generate a mock model ACPO-PLU ML framework

The generated model is not a neural net - it is just a tf.function with the
correct input and output parameters.
By construction, the mock model will always average the input values and return as output.
"""
import os
import importlib.util
import sys
import tensorflow as tf

POLICY_DECISION_LABEL = 'unrolling_decision'
POLICY_OUTPUT_SPEC = """
[
    {
        "logging_name": "unrolling_decision",
        "tensor_spec": {
            "name": "StatefulPartitionedCall",
            "port": 0,
            "type": "float64_t",
            "shape": [
                1
            ]
        }
    }
]
"""

def get_input_signature():
  """Returns the list of features for LLVM ACPO-unrolling."""
  inputs = [
      tf.TensorSpec(dtype=tf.float32, shape=(1,10), name=key) for key in [
        'PLU_features', #The input features with a shape of (1,10)
      ]
  ]
  return inputs

def get_output_signature():
  return POLICY_DECISION_LABEL


def get_output_spec():
  return POLICY_OUTPUT_SPEC

def get_output_spec_path(path):
  return os.path.join(path, 'output_spec.json')

def build_mock_model(path, signature):
  """Build and save the mock model with the given signature"""
  module = tf.Module()
  module.var = tf.Variable(0.)

  def action(*inputs):
    s = tf.math.reduce_mean([tf.cast(x, tf.float32) for x in tf.nest.flatten(inputs)])
    return {signature['output']: float('0') + s + module.var}

  module.action = tf.function()(action)
  action = {'action': module.action.get_concrete_function(signature['inputs'])}
  tf.saved_model.save(module, path, signatures=action)

  output_spec_path = get_output_spec_path(path)
  with open(output_spec_path, 'w') as f:
    print(f'Writing output spec to {output_spec_path}.')
    f.write(signature['output_spec'])

def get_signature():
  return {
      'inputs': get_input_signature(),
      'output': get_output_signature(),
      'output_spec': get_output_spec()
  }

def main(argv):
  assert len(argv) == 2
  model_path = argv[1]

  print(f'Output model to: [{argv[1]}]')
  signature = get_signature()
  build_mock_model(model_path, signature)

if __name__ == '__main__':
  main(sys.argv)
