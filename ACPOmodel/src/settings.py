import argparse
import json
import os

import yaml

from losses import loss_dict

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in {'yes', 'true', 't', 'y', '1'}:
        return True
    elif v.lower() in {'no', 'false', 'f', 'n', '0'}:
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

# Training settings
parser = argparse.ArgumentParser(description='Huawei ACPOModel Supported Arguments')
group = parser.add_mutually_exclusive_group()
group.add_argument('-v', '--verbose', action='store_true')
group.add_argument('-q', '--quite', action='store_true')
parser.add_argument('-o', '--output', action='store_true', help='Output the result to a file')

parser.add_argument('--load-json', action='store_true', default=True)
parser.add_argument('--save-json', action='store_true', default=False)

parser.add_argument('--root-path', type=str, required=True)
parser.add_argument('--user-config', type=str, default="", required=True,
                    help='the path to the user configuration file.')

parser.add_argument('--data-path', type=str, default="data")
parser.add_argument('--data-dir', type=str, default='./data/',
                    help='path of directory to access training files')
parser.add_argument('--log-dir', type=str, default='standalone',
                    help='Path of directory to save training logs, stats, plots, and model')
parser.add_argument('--work-dir', type=str, default='./work/',
                    help='Work directory holds temp folder for training dataset and test folder for validation dataset.')
parser.add_argument('--erase-work', action='store_true', default=False,
                    help="Empty work directory after training.")
parser.add_argument('--build-data', action='store_true', default=False,
                    help="If we want to use the current data in work directory.")
parser.add_argument('--save-model', action='store_true', default=False,
                    help='For Saving the current Model')

parser.add_argument('--prune-key', type=str, default=None, 
                    help="The key/feature we use to prune data.")
parser.add_argument('--prune-value', type=list, default=None, 
                    help="Threshold which we will use to prune data.")
parser.add_argument('--unique-key', type=list, default=None, 
                    help="Unique key(s) to dataset which determine duplicates.")
parser.add_argument('--class-key', type=list, default='Classes',
                    help="key(s)/feature(s) used to convert create classes.")
parser.add_argument('--num-classes', type=int, default=1,
                    help="Number of possible classes which can be derived from the dataset itself.")
parser.add_argument('--x-col-start', type=int, default=0,
                    help="Start index of the feature columns in the dataset.")
parser.add_argument('--x-col-end', type=int, default=-2,
                    help="End index of the feature columns in the dataset.")
parser.add_argument('--y-col', type=int, default=-1,
                    help="The index of the class column in the dataset.")

parser.add_argument('--incl-noinline-features', action='store_true', default=False,
                    help='For each caller and inline configuration, include features of non-inlined version of caller')
parser.add_argument('--feature-select', action='store_true', default=False,
                    help='Use a feature selection method')
parser.add_argument('--feature-scale', choices=['off', 'separate', 'same'], default='same',
                    help='Whether to use off|same|separate feature scaling for training and testing')
parser.add_argument('--pca', action='store_true', default=False,
                    help='Apply PCA')
parser.add_argument('--plot', action='store_true', default=False,
                    help='Plot the prediction accuracy')
parser.add_argument('--debug', type=str, nargs='+', default='',
                    help='comma separated values or features to debug. Choose one or more of: [correlation,csv,heatmap,pairplot,corrplot,gen-speedup,drop-no-inline,drop-functions-without-speedup,drop-duplicates,pruning-outliers,drop-uncorrelated-features,final,pca,feature-select,feature-scaling]')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--dry-run', action='store_true', default=False,
                    help='quickly check a single pass')

parser.add_argument('--algorithm', choices=['nn', 'dt'], default='nn',
                    help='machine learning algorithm to use')
parser.add_argument('--task', choices=['regression', 'classification'],
                    default='classification',
                    help='Type of ML task to apply to')

parser.add_argument('--cv', choices=['loocv', 'standalone', 'kfold'], default='standalone',
                    help='Cross validation method to choose')
parser.add_argument('--folds', type=int, default=10,
                    help='Number of k-fold for cross validation method')

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 16)')
parser.add_argument('--test-batch-size', type=int, default=16, metavar='N',
                    help='input batch size for testing (default: 16 )')
parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='number of epochs to train (default: 5)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--lr-step', type=int, default=1, metavar='LR',
                    help='learning rate step (default: 1)')
parser.add_argument('--gamma', type=float, default=0.1, metavar='M',
                    help='Learning rate step gamma (default: 0.1)')
parser.add_argument('--loss', choices=loss_dict.keys(), default='cross_entropy',
                    metavar='LOSS_FN',
                    help='Loss function (default: cross_entropy)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')


args = parser.parse_args()


def sync_config(user_path:str=None):
    """This method update configuration based on the user configuration

    Args:
        user_path (json or yaml): override the default configuration by user
            Example:
                user_path = "user_config.json"
                user_path = "user_config.yaml"
    Returns:
        argparse: the updated configuration
    """
    if not user_path:
        user_path = args.user_config

    config = None
    if args.load_json:
        with open(user_path, 'r') as file:
            user_args = argparse.Namespace()
            
            if user_path[-4:] =="json":
                data = json.load(file)
            else:
                data = yaml.safe_load(file)
            
            user_args.__dict__.update(data)
            config = parser.parse_args(namespace=user_args)

    if args.save_json:
        user_path_new = os.path.join(args.user_path + "_new" + ".json")
        with open(user_path_new, 'w') as file:
            json.dump(vars(config), file, indent=4)
            file.close()

    return config if config else args


def yaml_to_json(source:str, destination:str=None) -> None:

    destination = destination if destination else source

    with open(source + ".yaml", 'r') as yml:
        data = yaml.safe_load(yml)

    with open(destination + ".json", 'w') as js:
        json.dump(data, js)


def json_to_yaml(source:str, destination:str=None) -> None:

    destination = destination if destination else source

    with open(source + ".json", 'r') as js:
        data = json.load(js)

    with open(destination + ".yaml", 'w') as yml:
        yaml.safe_dump(data, yml, allow_unicode=True)

