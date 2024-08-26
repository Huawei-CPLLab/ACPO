#===- training_data_generator.py - ACPO Training Data Generator    ------===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2021-2023. Huawei Technologies Co., Ltd. All rights reserved.
#
#===----------------------------------------------------------------------===//
import argparse
import csv
import os
import shlex
import shutil
import subprocess
import sys

def write_csv_data(output_file: str, iter: str, new_runtime: str, speedup: str):
     with open(output_file, 'a') as file:
        writer = csv.writer(file)
        writer.writerow([iter, new_runtime, speedup])

def run_benchmark(compile_dir: str, make_target: str, clean_command: str,
                  autotuner: str, run_command: str, unnamed_var_prefix: str,
                  output_file_path: str, ir_file_dir: str, ir_dump_mode: str, iter: int):
    print(f"Iteration: {iter}")
    # Clean previous build
    os.chdir(compile_dir)
    subprocess.run(shlex.split(clean_command), stdout=subprocess.DEVNULL)

    # Compile with -fautotune
    # The data is appended to the existing ACPO model file
    if not run_baseline:
        cp = subprocess.run(shlex.split(f"make {make_target} HUAWEIFLAGS=\"-O3 -fautotune " \
                        f"-mllvm -IR-file-directory={ir_file_dir} " \
                        f"-mllvm -unnamed-var-prefix={unnamed_var_prefix} " \
                        f"-mllvm -ACPO-model-file={output_file_path} "
                        f"-mllvm -loop-unroll-dump-mode={ir_dump_mode}\""),
                        stdout=subprocess.PIPE)
        cp.check_returncode()
    else:
        cp = subprocess.run(shlex.split(f"make {make_target} HUAWEIFLAGS=\"-O3\""),
                        stdout=subprocess.PIPE)
        cp.check_returncode()

    cp = subprocess.run(shlex.split("/usr/local/bin/flush-cache"))
    cp.check_returncode()
    cp = subprocess.run(shlex.split("sleep 1"))
    cp.check_returncode()

    cp = subprocess.run(shlex.split(run_command), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    cp.check_returncode()

    # Extract the runtime from the output
    realtime = subprocess.run(["grep", "time elapsed"], input=cp.stderr,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    runtime_str = subprocess.run(["awk", "{print $1}"], input=realtime.stdout,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    runtime = float(runtime_str.stdout)

    return runtime

def autotune_generate(compile_dir: str, make_target: str, clean_command: str, autotuner: str,
                      code_region: str, pass_filter: str, search_space: str):
    print("Generating autotuning opportunity")

    # Clean previous build
    os.chdir(compile_dir)
    subprocess.run(shlex.split(clean_command), stdout=subprocess.DEVNULL)

    # Compile with -fautotune-generate to generate tuning opportunities
    cp = subprocess.run(shlex.split(f"make {make_target} HUAWEIFLAGS=\"-O3 " \
                        f"-fautotune-generate{code_region} {pass_filter}\""),
                        stdout=subprocess.PIPE)
    cp.check_returncode()

    print("Finish generating autotuning opportunity")

    cp = subprocess.run(shlex.split(f"{autotuner} minimize {search_space}"))
    cp.check_returncode()

# return: 1) Directory containing the benchmark makefile
#         2) Run command for the benchmark
#         3) Make target and path to bin directory containing clang/clang++
#         4) Clean command for the benchmark
def get_benchmark_info(benchmark_name: str, benchmark_dir: str, llvm_dir: str):
    measure_command = "numactl --localalloc --physcpubind=8 perf stat -r3"
    if benchmark_name == "amg":
        return os.path.join(benchmark_dir, "coral-2/AMG"), \
               f"{measure_command} ./test/run-autotuner.sh 2>&1 1>/dev/null", \
               "CLANG=" + os.path.join(llvm_dir, "clang"), \
               "make clean"
    elif benchmark_name == "coremark":
        return os.path.join(benchmark_dir, "coremark"), \
               f"{measure_command} ./coremark.exe 0x0 0x0 0x66 0 7 1 2000", \
               "compile CC=" + os.path.join(llvm_dir, "clang"), \
               "make clean"
    elif "stride" in benchmark_name:
        program_name=benchmark_name.split(" ", 1)[1] + ".Opt"
        return os.path.join(benchmark_dir, "coral-2/STRIDE/src"), \
               f"{measure_command} ./{program_name}", \
               f"{program_name} CC=" + os.path.join(llvm_dir, "clang"), \
               "make clean"

# This is the main procedure. First, compile with -fautotune-generate to generate
# opportunities. Then autotune the benchmarks for a pre-defined number of iterations.
# In each iteration, record the data and IR files for code regions.
def collect_training_data(benchmark_name: str, benchmark_dir: str, llvm_dir: str,
                          autotuner:str, code_region: str, pass_filter: str,
                          search_space: str, num_iter: int, output_file: str,
                          unnamed_var_prefix: str, ir_dump_mode: str):
    compile_dir, run_command, make_target, clean_command = \
                get_benchmark_info(benchmark_name, benchmark_dir, llvm_dir)
    if not os.path.exists(compile_dir):
        print(f"{benchmark_name}: benchmark not found")
        sys.exit(1)

    if not os.environ.get("AUTOTUNE_DATADIR"):
        print("AUTOTUNE_DATADIR environment variable is not set. "
               "Using the default 'autotune_datadir' directory")
        os.environ["AUTOTUNE_DATADIR"] = os.path.join(compile_dir, "autotune_datadir")

    # Summary data file is stored in AUTOTUNE_DATADIR
    os.makedirs(name=os.path.expandvars("$AUTOTUNE_DATADIR"), exist_ok=True)
    output_file_path=os.path.join(os.path.expandvars("$AUTOTUNE_DATADIR"), output_file)
    autotune_generate(compile_dir, make_target, clean_command, autotuner,
                      code_region, pass_filter, search_space)

    # Create directories to save config and data files
    autotune_datadir=os.path.expandvars("$AUTOTUNE_DATADIR")
    config_file_save_path=os.path.join(autotune_datadir, "config")
    os.makedirs(name=config_file_save_path, exist_ok=True)
    data_file_save_path=os.path.join(autotune_datadir, "data")
    os.makedirs(name=data_file_save_path, exist_ok=True)
    # File to save runtime
    write_csv_data(os.path.join(compile_dir, "runtime.csv"), "Iteration", "Runtime", "Speedup")

    ir_file_dir=os.path.join(os.path.expandvars("$AUTOTUNE_DATADIR"), "ir")
    base_runtime = run_benchmark(compile_dir, make_target, clean_command,
                                    autotuner, run_command, unnamed_var_prefix,
                                    output_file_path, ir_file_dir, ir_dump_mode, iter=0, run_baseline=True)
    print(f"Baseline runtime: {base_runtime}")

    for i in range(0, num_iter):
        new_runtime = run_benchmark(compile_dir, make_target, clean_command,
                                    autotuner, run_command, unnamed_var_prefix,
                                    output_file_path, ir_file_dir, ir_dump_mode, i, run_baseline=False)
        #write_csv_data(os.path.join(compile_dir, output_file_path), f"{new_runtime}", f"Finish iteration {i}")
        # Save this iteration config and data files, and record runtime
        shutil.copy(os.path.join(autotune_datadir, "config.yaml"), os.path.join(config_file_save_path, f"config-{i}.yaml"))
        shutil.move(output_file_path, os.path.join(data_file_save_path, f"data-{i}.csv"))
        write_csv_data(os.path.join(compile_dir, "runtime.csv"), f"{i}", f"{new_runtime}", f"{base_runtime/new_runtime}")
        cp = subprocess.run(shlex.split(f"{autotuner} feedback {new_runtime}"), stdout=subprocess.DEVNULL)
        cp.check_returncode()

    print("Finalize tuning and generate the optimal compiler configuration")
    cp = subprocess.run(shlex.split(f"{autotuner} finalize"))
    cp.check_returncode()

def main():
    home_dir = os.path.expandvars("$HOME")
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark",
        choices=["amg", "coremark", "stride strid3c", "stride cachec", "stride vecopc"],
        help="The benchmark to be run")
    parser.add_argument("--benchmark_dir", type=str, default=os.path.join(home_dir, "benchmarks"),
        help="Path to directory containing the benchmarks (Default: $HOME/benchmarks/).")
    parser.add_argument("--llvm_dir", type=str, default=os.path.join(home_dir, "bisheng/bin"),
        help="Path to directory containing LLVM binaries (Default: $HOME/bisheng/bin).")
    parser.add_argument("--llvm_autotune", type=str, default=os.path.join(home_dir, "autotuner/bin/llvm-autotune"),
        help="Path to llvm-autotune (Default: $HOME/autotuner/bin/llvm-autotune).")
    parser.add_argument("--search_space", type=str,
        help="Specify the search space for Autotuner (Default: default Autotuner search space).")
    parser.add_argument("--pass_filter", type=str,
        help="Only dump training data for the specified pass (Default: no filter).")
    parser.add_argument("--code_region_filter", type=str,
        help="Only dump training data for the specified code region (Default: default Autotuner setting).")
    parser.add_argument("-o", "--output", default="acpo-data.csv",
        help="Summary csv file name (Default: acpo-data.csv).")
    parser.add_argument("-i", "--iteration", type=int, default=1000,
        help="Number of iterations to run for data collection (Default: 1000).")
    parser.add_argument("--unnamed_var_prefix", type=str, default="acpo",
        help="Prefix added to unnamed variables (Default: acpo).")
    parser.add_argument("--ir_dump_mode", type=str, default="before,loop",
        help="Specify IR dump mode (Default: before,loop).")
    args = parser.parse_args()

    if not os.path.isfile(os.path.join(args.llvm_dir, "clang")):
        print(f"{args.llvm_dir}: clang/clang++ not found")
        sys.exit(1)

    if not os.path.isfile(args.llvm_autotune):
        print(f"{args.llvm_autotune}: llvm_autotune not found")
        sys.exit(1)

    search_space = ""
    if args.search_space is not None:
        search_space_file = os.path.abspath(args.search_space)
        search_space = f"--search-space={search_space_file}"

    pass_filter = ""
    if args.pass_filter is not None:
        pass_filter = f"-mllvm -auto-tuning-pass-filter={args.pass_filter}"

    code_region_filter = ""
    if args.code_region_filter is not None:
        code_region_filter = f"={args.code_region_filter}"

    collect_training_data(args.benchmark, args.benchmark_dir, args.llvm_dir,
        args.llvm_autotune, code_region_filter, pass_filter,
        search_space, args.iteration, args.output,
        args.unnamed_var_prefix, args.ir_dump_mode)

if __name__ == "__main__":
    main()
