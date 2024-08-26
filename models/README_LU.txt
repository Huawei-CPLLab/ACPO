=== ACPO partial Loop Unrolling Model ===

(1) Run the model generator script to generate a dummy PLU model which takes 10 input features and output the average:

    python3 gen-unroll-test-model_v1.py plu-dummy/

(2) Use commanldline TF Saved_Model tool to inspect the input/output of the model:

    saved_model_cli show --all --dir plu-dummy/

'''
MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

signature_def['__saved_model_init_op']:
  The given SavedModel SignatureDef contains the following input(s):
  The given SavedModel SignatureDef contains the following output(s):
    outputs['__saved_model_init_op'] tensor_info:
        dtype: DT_INVALID
        shape: unknown_rank
        name: NoOp
  Method name is:

signature_def['action']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['PLU_features'] tensor_info:
        dtype: DT_FLOAT
        shape: (1, 10)
        name: action_PLU_features:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['unrolling_decision'] tensor_info:
        dtype: DT_FLOAT
        shape: ()
        name: StatefulPartitionedCall:0
  Method name is: tensorflow/serving/predict

Defined Functions:
  Function Name: 'action'
    Option #1
      Callable with:
        Argument #1
          DType: list
          Value: [TensorSpec(shape=(1, 10), dtype=tf.float32, name='PLU_features')]
'''

    As can be seen:
        -'action' is the signature_Def
        -'unrolling_decision' is the output

