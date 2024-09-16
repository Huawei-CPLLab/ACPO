#!/usr/bin/env python3
# ===- ACPO-dummy-inliner-model-gen.py - Generate Dummy Inliner Model ------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===-----------------------------------------------------------------------===#

"""Generate a mock model ACPO Inline ML framework

The generated model is not a neural net - 
It is a tf.function with the correct input and output parameters.
By construction, the mock model will always return 1 as output (forces inlining).
"""
import os
import importlib.util
import sys
import tensorflow as tf

POLICY_DECISION_LABEL = 'output_0'
POLICY_OUTPUT_SPEC = """
[
    {
        "logging_name": "inlining_decision",
        "tensor_spec": {
            "name": "PartitionedCall",
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
  """Returns the list of features for LLVM ACPO Inliner."""
  inputs = [
      tf.TensorSpec(dtype=tf.float32, shape=(1,100), name=key) for key in [
        'input_1', #The input features with a shape of (1,100)
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
    return {signature['output']: tf.constant([0,1], dtype=tf.float32, shape=(1,2))}

  module.action = tf.function()(action)
  action = {'serving_default': module.action.get_concrete_function(signature['inputs'])}
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
