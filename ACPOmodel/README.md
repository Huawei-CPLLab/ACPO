# ACPO-model

ACPO-model is an end-to-end framework that replaces compiler optimization heuristics
with machine learning models. It comes together with Feature Quality Control (FQC) tools to
help analyze model relative to the training data in an effort to isolate useful features and
remove unnecessary ones.

# Getting Started

1. Set up environment

```sh
virtualenv .venv --prompt="(ml-opt) " --python=/usr/bin/python3.6
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run training script.
This will use the data under `./data` to perform a training and validation.

```sh
cd ~/ACPO-model/src
python3 train.py
```

or create a Symlink inside the parent directory `ACPO-model`
```sh
ln -s src/fqc.py FeatureQC
ln -s src/train.py TrainModel
ln -s src/inference.py Inference
```

# Configuration
1. ACPO-Model supports CLI arguments that can be find in `settings.py`.
2. Other parameters can be defined as `user_config.json` which override default values.
3. It is possible to provide `user_config.yaml`.
4. There are some example `user_config.json` files for `train.py`, `inference.py`, and `fqc.py`. One can modify them to meet the needs.


### ACPO-Model Supported Arguments
```sh
Optional arguments:
  --help, -h            show this help message and exit

Set Paths
  --root-path           the path to the root directory of the project
  --user-config         the path to the user configuration file.

  --data-path           the path to the data directory
  --data-dir            path of directory to access training files
  --log-dir             Path of directory to save training logs, stats, plots, and model

user configuration options
  --load-json           load user configuration and synchronize with the current configuration.
  --save-json           save the current configuration

working setups
  --work-dir            Work directory holds temp folder for training dataset and test folder for validation dataset.
  --erase-work          Empty work directory after training.
  --build-data          If we want to use the current data in work directory.
  --save-model          For Saving the current Model

preprocessing settings
  --prune-key           The key/feature we use to prune data.
  --prune-value         Threshold which we will use to prune data.
  --unique-key          Unique key(s) to dataset which determine duplicates.
  --class-key           key(s)/feature(s) used to convert create classes.
  --num-classes         Number of possible classes which can be derived from the dataset itself.
  --x-col-start         Start index of the feature columns in the dataset.
  --x-col-end           End index of the feature columns in the dataset.
  --y-col               The index of the class column in the dataset.

  --incl-noinline-features 
                        For each caller and inline configuration, include features of non-inlined version of caller
  --feature-select      Use a feature selection method
  --feature-scale       Whether to use off/same/separate feature scaling for training and testing (default: same)
                        choices = [off, separate, same]

  --pca                 Apply PCA
  --plot                Plot the prediction accuracy
  --debug               comma separated values or features to debug.
                        Choose one or more of the following joined with +:
                        [correlation, corrplot, csv, 
                        drop-duplicates, drop-functions-without-speedup, drop-no-inline, drop-uncorrelated-features,
                        feature-scaling, feature-select, final, gen-speedup,
                        heatmap, pairplot, pca, pruning-outliers]

training configuration
  --no-cuda             disables CUDA training
  --dry-run             quickly check a single pass

  --algorithm           Machine learning algorithm to use (default: nn)
                        choices = [nn, dt]

  --task                Type of ML task to apply to (default: classification)
                        choices = [regression, classification]

  --cv                  Cross validation method to choose (default: standalone)
                        choices = [loocv, standalone, kfold]

hyper-parameters
  --folds               Number of k-fold for cross validation method (default: 10)
  --seed                random seed (default: 1)
  --batch-size          input batch size for training (default: 16)
  --test-batch-size     input batch size for testing (default: 16)
  --epochs              number of epochs to train (default: 5)
  --lr                  learning rate (default: 0.01)
  --lr-step             learning rate step (default: 1)
  --gamma               Learning rate step gamma (default: 0.1)
  --loss                Loss function (default: cross_entropy)
  --log-interval        how many batches to wait before logging training status (default: 100)
```

### Samples for `user_config` file for different purpose.

User configuration for the training purpose `train.py`:
- [Sample JSON settings](doc/config/user_train_config.json)
- [Sample YAML settings](doc/config/user_train_config.yaml)

User configuration for the inference purpose `inference.py`:
- [Sample JSON settings](doc/config/user_inference_config.json)
- [Sample YAML settings](doc/config/user_inference_config.yaml)

User configuration for the training purpose `fqc.py`:
- [Sample JSON settings](doc/config/user_fqc_config.json)
- [Sample YAML settings](doc/config/user_fqc_config.yaml)
