# AI-Enabled Continous Program Optimization (ACPO) Framework

The ACPO framework is designed to easily integrate ML models within a compiler framework and provide useful
tools for training and analysis of ML models for use in a compiler. It comes together with examples of different
models that were deployed in an LLVM-based compiler.

At a high level, ACPO separates the compiler from ML by providing a simple abstraction layer where the compiler communicates
with an ML framework via an predefined interface and the ML framework runs inference on models the compiler requires. The models
are trained on data collected from compiler runs and thus can easily help substitute for profitability analysis that are generally hand-written.

In this project, there are a couple of key contributions:
1. ACPO-model framework with training, inference and feature quality control (FQC) tools to simpify inclusion of ML models in a compiler
2. A set of script to enable ML framework to run as a parallel process to the compiler using named pipes
3. A set of example models

# Getting Started

Download the project into an appropriate folder in your compiler project, where it can be easily included in a build process that suits your needs. Generally,
ACPO only requires to be visible to a project which references interfaces provided by ACPO and since it is written in Python binary generation is optional.

Each section of this repository contains requirements.txt files that specify python packages you will need to run the associated scripts with. Please ensure 
that you have the appropriate packages installed to have ACPO behave as intended.

# Contributions

The ACPO framework is an evolving framework, part of a larger effort to make enable compilers with AI technology. We welcome contributions that make this
framework better, faster and and larger in scope. This include providing models for various passes for community to use and build on. Please note that companion
projects, for us specifically related to LLVM-based infrastructure, provide feature collection methods and interfaces to leverage ACPO in an easy-to-use way.

Feel free to reach out to us and contribute to help us make data-driven compilers with AI/ML capabilities.