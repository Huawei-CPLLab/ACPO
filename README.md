# AI-Enabled Continuous Program Optimization (ACPO) Framework

The ACPO framework is designed to easily integrate ML models within a compiler framework and provide useful
tools for training and analysis of ML models for use in a compiler. It comes together with examples of different
models that were deployed in an LLVM-based compiler.

At a high level, ACPO separates the compiler from ML by providing a simple abstraction layer where the compiler communicates
with an ML framework via a predefined interface and the ML framework runs inference on models the compiler requires. The models
are trained on data collected from compiler runs and thus can easily help substitute for profitability analyses that are generally hand-written.

In this project, there are a couple of key contributions:
1. ACPO-model framework with training, inference, and feature quality control (FQC) tools to simplify the inclusion of ML models in a compiler
2. A set of scripts to enable the ML framework to run as a parallel process to the compiler using named pipes
3. A set of example models

# Getting Started

Download/clone the project into an appropriate folder in your compiler project, i.e., [OpenEuler](https://gitee.com/openeuler/llvm-project/tree/dev_17.0.6/), where it can be easily included in a build process that suits your needs. Generally,
ACPO only requires to be visible to a project that references interfaces provided by ACPO, and since it is written in Python, binary generation is optional.

Each section of this repository contains `requirements.txt` files that specify Python packages you will need to run the associated scripts with. Please ensure that you have the appropriate packages installed to have ACPO behave as intended. (Most importantly, `tensorflow==2.14.0` & python > 3.10)

# Deploying An ACPOModel with LLVM

Support compiler version: llvm `dev_17.0.6`, please clone from https://gitee.com/openeuler/llvm-project/tree/dev_17.0.6/ and place acpo folder under $LLVM folder.

1. Given each task, your model should have `X` input and `Y` output values. This should be reflected in the structure of the model, and that should also match your model spec file under `$LLVM/acpo/model-XXX.acpo`. Here is an example of `model-bw.acpo` file: 

```
ModelName=FI
Features={callee_BlockWithMultipleSuccecorsPerLoop, float32},{callee_PtrArgs, float32},{callee_MaxDomTreeLevel, float32},{callee_IsLinkOnceODR, float32},{callee_IsLocal, float32},{callee_Calls, float32},{callee_Blocks, float32},{callee_InitialSize, float32},{callee_MaxLoopDepth, float32},{callee_users, float32},{callee_InstructionPerBlock, float32},{callee_Loops, float32},{callee_conditionally_executed_blocks, float32},{callee_IsLinkOnce, float32},{callee_basic_block_count, float32},{callee_PtrCallee, float32},{callee_CallReturnPtr, float32},{callee_ConditionalBranch, float32},{callee_CBwithArg, float32},{callee_CallerHeight, float32},{callee_CallUsage, float32},{callee_IsRecursive, float32},{callee_NumCallsiteInLoop, float32},{callee_NumOfCallUsesInLoop, float32},{callee_EntryBlockFreq, float32},{callee_MaxCallsiteBlockFreq, float32},{callee_SuccessorPerBlock, float32},{callee_AvgVecInstr, float32},{callee_AvgNestedLoopLevel, float32},{callee_InstrPerLoop, float32},{caller_BlockWithMultipleSuccecorsPerLoop, float32},{caller_PtrArgs, float32},{caller_MaxDomTreeLevel, float32},{caller_IsLinkOnceODR, float32},{caller_IsLocal, float32},{caller_Calls, float32},{caller_Blocks, float32},{caller_InitialSize, float32},{caller_MaxLoopDepth, float32},{caller_users, float32},{caller_InstructionPerBlock, float32},{caller_Loops, float32},{caller_conditionally_executed_blocks, float32},{caller_IsLinkOnce, float32},{caller_basic_block_count, float32},{caller_PtrCallee, float32},{caller_CallReturnPtr, float32},{caller_ConditionalBranch, float32},{caller_CBwithArg, float32},{caller_CallerHeight, float32},{caller_CallUsage, float32},{caller_IsRecursive, float32},{caller_NumCallsiteInLoop, float32},{caller_NumOfCallUsesInLoop, float32},{caller_EntryBlockFreq, float32},{caller_MaxCallsiteBlockFreq, float32},{caller_SuccessorPerBlock, float32},{caller_AvgVecInstr, float32},{caller_AvgNestedLoopLevel, float32},{caller_InstrPerLoop, float32},{is_indirect, float32},{num_loops, float32},{opt_code, float32},{unsimplified_common_instructions, float32},{mandatory_only, float32},{switch_penalty, float32},{mandatory_kind, float32},{case_cluster_penalty, float32},{loop_level, float32},{jump_table_penalty, float32},{cost_estimate, float32},{indirect_call_penalty, float32},{nr_ctant_params, float32},{lowered_call_arg_setup, float32},{callsite_height, float32},{load_relative_intrinsic, float32},{block_freq, float32},{call_argument_setup, float32},{call_penalty, float32},{load_elimination, float32},{hot_callsite, float32},{sroa_losses, float32},{cold_callsite, float32},{sroa_savings, float32},{is_in_inner_loop, float32},{dead_blocks, float32},{is_must_tail, float32},{simplified_instructions, float32},{is_tail, float32},{constant_args, float32},{constant_offset_ptr_args, float32},{callsite_cost, float32},{cold_cc_penalty, float32},{last_call_to_static_bonus, float32},{is_multiple_blocks, float32},{nested_inlines, float32},{nested_inline_cost_estimate, float32},{threshold, float32},{node_count, float32},{edge_count, float32}
Outputs={FI-ShouldInline, int64}
Signature=serving_default
ModelDirectory=./models/pfi.pb
OutputKey=output_0
ModelInference=FIInference
# Above ModelInference need to be updated on python side
```


You can inspect some of the values above by running `saved_model_cli`:

`saved_model_cli --dir ~/DIR/TO/saved_model-pb/ show --all`

- `ModelName`: This one should match your ${MODELNAME}CompiledModel-${ARCH}.{o,h,inc}
- `Features`: It is a list of feature names and their types
- `Outputs`: The variable corresponding to the output value of the model and its type
- `Signature`: This is the signature of the model (default value is normally `seving_default` in TensorFlow)
- `ModelDirectory`: The location of the .pb file
- `OutputKey`: The output node of the trained model
- `ModelInference`: The C++ value of the inference
  
2. Add the following changes in LLVM:
   
- `llvm/lib/Analysis/ACPOMLInterface.cpp`
  Make sure ACPOMLInterface `llvm/lib/Analysis/ACPOMLInterface.cpp` has the ACPOModelRunner instance and the map in it:
```
#ifdef LLVM_HAVE_TF_AOT_FICOMPILEDMODEL
std::unique_ptr<ACPOModelRunner>
createFI(std::vector<std::pair<std::string, std::string>> Inputs,
         StringRef Decision) {
  // Context does not ever seem to be used in the model runner,
  // so for now just create an empty context object
  LLVMContext Ctx;
  return std::make_unique<FIModelRunner>(Ctx, Inputs, Decision);
}
#endif
```

```
#ifdef LLVM_HAVE_TF_AOT_FICOMPILEDMODEL
        {"FI", createFI},
#endif
```

- `llvm/include/llvm/Analysis/XXXModelRunner.h`
  Update each model's ModelRunner.h file with necessary info, such as  the scaler info (if you have used a data scaler during training, you need to update the `scale` and the `mean` variables):

```
from sklearn.feature_selection import *
import pickle as pk
sc = pk.load(open("acpo/models/ir2score.pb-IR2VEC/sc_ir2vec.pkl", "rb"))
>>> sc.
sc.copy                    sc.fit_transform(          sc.get_params(             sc.mean_                   sc.n_samples_seen_         sc.scale_                  sc.transform(              sc.with_mean
sc.fit(                    sc.get_feature_names_out(  sc.inverse_transform(      sc.n_features_in_          sc.partial_fit(            sc.set_params(             sc.var_                    sc.with_std
```

Also, the inference routine under `getModelresultI()`:
```
  // Outputs for this model are only int so we only need to override this
  // method
  int getModelResultI(std::string OutputName) override {
    if (OutputName == "FI-ShouldInline") {
      int Classes[] = {0, 1};
      void *ResultUntyped = CompiledModel->result_data(0);
      float *Result = reinterpret_cast<float *>(ResultUntyped);
      float Max = Result[0];
      int MaxClass = 0;
      for (size_t I = 0; I < sizeof(Classes) / sizeof(int); ++I) {
        if (Result[I] > Max) {
          Max = Result[I];
          MaxClass = I;
        }
      }
      LLVM_DEBUG(dbgs() << "THE DECISION IS: " << Classes[MaxClass] << "\n");
      return Classes[MaxClass];
    }
    assert(false && "ModelRunner received invalid result name");
  }
};
```

3. Generate precompiled (c++) AOT model files so that during LLVM build, the scripts can include them in the build folder and link them with clang. Note that if you update models, you need a rebuild so changes are getting effected (this is not necessary when using the Python interface).

- Save `.pb` file as `.h` and `.o` files (Optional step to generate override files for `-o ARCH`). 

`saved_model_cli aot_compile_cpu --multithreading false --dir /DIR/TO/BW/MODEL.pb/ --tag_set serve --signature_def_key serving_default --output_prefix FICompiledModel --cpp_class llvm::FICompiledModel`

**NOTE:** If you want to generate aarch64 format .h/.o file, please append `--target_triple aarch64-unknown-linux-gnu` after the command above. Generate aarch64 format .h/.o files on an aarch64 server is suggested, so as for x86 format files. 

You should now have 3 files: `FICompiledModel.h`, `FICompiledModel.o`, `FICompiledModel_metadata.o` (and a makefile, but we can ignore that). Take these 3 files and move them into the override folder `acpo/overrides` and append them with the system you're working with, -AARCH64 or -X86 (E.g., FICompiledModel-AARCH64.h, FICompiledModel-AARCH64.o, FICompiledModel_metadata-AARCH64.o)

4. Finally, rebuild LLVM using the build.sh and make sure your generated AOT (precompiled models) are now placed under `./build/lib/Analysis`, and also they are linked with `clang` by a full rebuild of LLVM:

```
ls ./build/lib/Analysis/
AI4CFHCompiledModel.h  AI4CFHCompiledModel.o  AI4CMEMOPCompiledModel.h  AI4CMEMOPCompiledModel.o  BWCompiledModel.h  BWCompiledModel.o  CMakeFiles  cmake_install.cmake  FICompiledModel.h  FICompiledModel.o
```


# Reference
If you use any of the materials in this project, i.e., codes, provided ready-to-go datasets, methodology, model, etc., you should cite this work:
* ACPO: AI-Enabled Compiler Framework (arXiv: https://arxiv.org/abs/2312.09982)
* Open Euler Source Code: https://gitee.com/openeuler/llvm-project

```
@inproceedings{Ashouri_2024,
   title={Work-in-Progress:ACPO: An AI-Enabled Compiler Framework},
   url={http://dx.doi.org/10.1109/CASES60062.2024.00011},
   DOI={10.1109/cases60062.2024.00011},
   booktitle={2024 International Conference on Compilers, Architecture, and Synthesis for Embedded Systems (CASES)},
   publisher={IEEE},
   author={Ashouri, Amir H. and Manzoor, Muhammad Asif and Vu, Minh and Zhang, Raymond and Wang, Ziwen and Zhang, Angel and Chan, Bryan and Czajkowski, Tomasz S. and Gao, Yaoqing},
   year={2024},
   month=sep, pages={21–21} }
```

# Contributions

The ACPO framework is an evolving framework, part of a larger effort to enable compilers with AI technology. We welcome contributions that make this framework better, faster, and wider in scope. This includes providing models for various passes for the community to use and build on. Please note that the companion projects, for us specifically related to LLVM-based infrastructure, provide feature collection methods and interfaces to leverage ACPO in an easy-to-use way.

Feel free to reach out to us and contribute to help us make data-driven compilers with AI/ML capabilities.
