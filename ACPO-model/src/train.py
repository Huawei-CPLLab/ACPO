# -------------------------- Import required libraries -------------------------
from __future__ import print_function

import glob
import os
import time
import subprocess

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import tree
from sklearn.model_selection import *
from sklearn.preprocessing import *
from torch.optim.lr_scheduler import StepLR

from csv_process import *  # CSV data preprocessing
from losses import loss_dict  # Loss dictionary
from models import *  # Model selection settings
from settings import *  # Training settings
from utils import *  # Utility and metrics

# ------------------------- GPU/CPU Pytorch selection --------------------------

# pytorch configuration to use cuda if GPU available or CPU for computation
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")

loss_fn = loss_dict[args.loss]   # assign loss function from given arguments


# ----------------------------- Learning Process -------------------------------
def trainClassification(args, model, device, train_loader, optimizer, loss_func, epoch, log_dir):

    actual_list, pred_list, loss_list, grad_list = [], [], [], []

    model.train()
    metrics = pd.DataFrame(columns=["correct", "total", "accuracy"])

    for batch_idx, (data, target) in enumerate(train_loader):

        data, label = data.to(device), target.to(device)
        output      = model(data)
        loss        = loss_func(output, label.squeeze(1))
        loss.backward()
        grad        = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        optimizer.step()
        optimizer.zero_grad()

        if batch_idx % args.log_interval == 0:

            print(f"Train Epoch: {epoch} "
                  f"[{batch_idx * len(data)}/{len(train_loader.dataset)} "
                  f"({100. * batch_idx / len(train_loader):.0f}%)] "
                  f"Loss: {loss.item():.6f}"
                )

            if args.dry_run:
                break

        metrics.loc[batch_idx] = calc_classification_accuracy(label, output)

        actual_list = np.append(actual_list, target.data.detach().cpu().numpy())
        pred_list   = np.append(  pred_list, output.data.detach().cpu().numpy())
        loss_list   = np.append(  loss_list,   loss.data.detach().cpu().numpy())
        grad_list   = np.append(  grad_list,   grad.data.detach().cpu().numpy())


    # TODO: Plot per epoch 
    if args.plot:
        plot(actual_list, pred_list, log_dir, "train")
        plotloss(loss_list,  log_dir, "Training Loss Plot")
        plotgrad(grad_list,  log_dir, "Gradient Norm Plot")


    print("\nTrain:")
    print(f"Number of batches: {len(train_loader)}. "
          f"Number of samples: {len(train_loader.dataset)}")

    correct  = metrics["correct"].sum()
    total    = metrics["total"].sum()
    accuracy = 100 * correct/total

    print(f"{correct} out of {total} training samples classified correctly \n")
    print(f"Classification Accuracy: {accuracy:.3f} \n")

    return accuracy, metrics


def testClassification(args, model, device, test_loader, log_dir, batch_log=False):

    # test_loss, test_loss_mse = 0, 0
    # test_loss_batch, test_loss_batch_mse = 0, 0
    # pred_mse = 0
    actual_list, pred_list = [], []

    model.eval()
    metrics = pd.DataFrame(columns=["correct", "total", "accuracy"])

    with torch.no_grad():

        for batch_idx, (data, target) in enumerate(test_loader):

            data, target = data.to(device), target.to(device)
            output       = model(data)

            if args.num_classes > 5:
                top_n = top5_metric(data, target, output)

                # calculating metrics for evaluating the model
                top_accuracy = [100 * top_n[i].item()/len(data) for i in range(5)]
                if batch_log:
                    print(f"Top-n accuracy: {top_accuracy}")


            #print (output.data, target.data, output.data - target.data)  # Actual vs. Predictions
            actual_list = np.append(actual_list, target.data.detach().cpu().numpy())
            pred_list   = np.append(  pred_list, output.data.detach().cpu().numpy())

            metrics.loc[batch_idx] = calc_classification_accuracy(target, output)

    if args.plot:
        plot(actual_list, pred_list, log_dir, "test")


    print("\nTest:")
    print(f"Number of batches: {len(test_loader)}. "
          f"Number of samples: {len(test_loader.dataset)}")

    correct = metrics["correct"].sum()
    total   = metrics["total"].sum()
    accuracy = 100 * correct/total

    print(f"{correct} out of {total} test samples classified correctly \n")
    print(f"Classification Accuracy: {accuracy:.3f} \n")

    return accuracy, metrics


def trainRegression(args, model, device, train_loader, optimizer, loss_func, epoch, log_dir):

    actual_list, pred_list, loss_list, grad_list = [], [], [], []

    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), (target.to(device)).type(torch.float)
        output       = model(data)
        loss         = loss_func(output, target)
        loss.backward()
        grad         = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        optimizer.step()
        optimizer.zero_grad()

        if batch_idx % args.log_interval == 0:

            print(f"Train Epoch: {epoch} "
                  f"[{batch_idx * len(data)}/{len(train_loader.dataset)} "
                  f"({100. * batch_idx / len(train_loader):.0f}%)] "
                  f"Loss: {loss.item():.6f}"
                )

            if args.dry_run:
                break

        actual_list = np.append(actual_list, target.data.detach().cpu().numpy())
        pred_list   = np.append(  pred_list, output.data.detach().cpu().numpy())
        loss_list   = np.append(  loss_list,   loss.data.detach().cpu().numpy())
        grad_list   = np.append(  grad_list,   grad.data.detach().cpu().numpy())


    #TODO: Plot per epoch 
    if args.plot:
        plot(actual_list, pred_list, log_dir, "train")
        plotloss(loss_list,  log_dir, "Training Loss Plot")
        plotgrad(grad_list,  log_dir, "Gradient Norm Plot")


    metrics = calc_metrics(actual_list, pred_list, "train_")

    print("\nTrain:")
    print(f"Number of batches: {len(train_loader)}. "
          f"Number of samples: {len(train_loader.dataset)}")

    print(f"Absolute Error: "
          f"Mean: {metrics['train_mae']:.3f}, "
          f"Mean Percentage: {100*metrics['train_mape']:.0f}%")

    print(f"Square Error: Mean: {metrics['train_mse']:.3f}\n")

    accuracy = -metrics['train_mse']
    return accuracy, metrics


def testRegression(args, model, device, test_loader, log_dir):

    # test_loss, test_loss_mse = 0, 0
    # test_loss_batch, test_loss_batch_mse = 0, 0
    # pred, pred_mse = 0, 0
    actual_list, pred_list = [], []

    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            #print (output.data, target.data, output.data - target.data)  # Actual vs. Predictions
            actual_list = np.append(actual_list, target.data.detach().cpu().numpy())
            pred_list = np.append(pred_list, output.data.detach().cpu().numpy())


    if args.plot:
        plot(actual_list, pred_list, log_dir, "test")


    metrics = calc_metrics(actual_list, pred_list, "test_")

    print("\nTest:")
    print(f"Number of batches: {len(test_loader)}. "
          f"Number of samples: {len(test_loader.dataset)}")

    print(f"Absolute Error: "
          f"Mean: {metrics['test_mae']:.3f}, "
          f"Mean Percentage: {100*metrics['test_mape']:.0f}%")

    print(f"Square Error: Mean: {metrics['test_mse']:.3f}\n")

    accuracy = -metrics['test_mse']
    return accuracy, metrics

def learn(args, train_data_path:str, test_data_path:str, log_dir:str, train_kwargs:dict, test_kwargs:dict):
    """This function fit a model for the given training data.

    Args:
        train_data_path (str): the path to the training data.
        test_data_path (str): the path to the test data.
        log_dir (str): the path to the save the model.
        train_kwargs (dict): parameters to pass for the training purposes.
        test_kwargs (dict): parameters to pass for the test purposes.

    Returns:
        csv: accuracy of the trained model.
    """

    print ("\nImporting Train Data:")
    train_dataset = CSVDataset(train_data_path,
                            args.incl_noinline_features,
                            feature_scale=args.feature_scale!='off',
                            feature_transform_pca=args.pca,
                            feature_select=args.feature_select,
                            mode='train',
                            x_col_start=args.x_col_start,
                            x_col_end=args.x_col_end,
                            y_col=args.y_col,
                            log_dir=args.log_dir,
                            debug=args.debug,
                            )

    print ("\nImporting Test Data:")
    test_dataset = CSVDataset(test_data_path,
                            args.incl_noinline_features,
                            feature_scale=train_dataset.get_feature_scaler() if\
                                        args.feature_scale=='same' else args.feature_scale!='off',
                            feature_transform_pca=train_dataset.get_feature_pca(),
                            feature_select=train_dataset.get_feature_selector(),
                            x_col_start=args.x_col_start,
                            x_col_end=args.x_col_end,
                            y_col=args.y_col,
                            mode='test',
                            log_dir=args.log_dir,
                            debug=args.debug,
                            )

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader  = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    num_features = train_dataset.num_features()
    # num_classes  = train_dataset.num_classes()
    print(f"# of features: {num_features} \n"
          f"# of classes: {args.num_classes} \n")

    train_log = CSVLogger(os.path.join(log_dir, "log_train.csv"))
    test_log  = CSVLogger(os.path.join(log_dir, "log_test.csv" ))
    csv_log   = CSVLogger(os.path.join(log_dir, "log.csv"      ))

    best_epoch   = None
    test_metrics = dict()

    loss_fn = loss_dict[args.loss]   # assign loss function from given arguments
    if args.algorithm == 'nn':

        if args.task == 'regression':
            model = Net(num_features).to(device)

        if args.task == 'classification':
            model = Net2(num_features, args.num_classes).to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=args.gamma)


        for epoch in range(1, args.epochs + 1):

            if args.task == 'regression':
                train_epoch_accuracy, train_epoch_log = trainRegression(args, model, device, train_loader, optimizer, loss_fn, epoch, log_dir)
                test_epoch_accuracy, test_epoch_log  = testRegression(args, model, device, test_loader, log_dir)

            if args.task == 'classification':
                train_epoch_accuracy, train_epoch_log = trainClassification(args, model, device, train_loader, optimizer, loss_fn, epoch, log_dir)
                test_epoch_accuracy, test_epoch_log   = testClassification(args, model, device, test_loader, log_dir)

            scheduler.step()

            train_log.append_df(epoch, train_epoch_log)
            test_log.append_df(epoch, test_epoch_log)

            csv_log.append(epoch, train_epoch_accuracy, test_epoch_accuracy)

            # need to be consistent with the new changes
            is_best = is_better(test_epoch_accuracy, best_epoch)
            if is_best:
                best_epoch = test_epoch_accuracy

                if args.save_model:
                    print("Start saving models")
                    if args.save_model_type == 'torch':
                        scrip_model = torch.jit.script(model)
                        scrip_model.save(os.path.join(log_dir, args.model["model_file"]))
                    elif args.save_model_type == 'tensorflow':
                        torch.save(model.state_dict(), os.path.join(log_dir, "plu.pt"))
                        save_pb(args, model, log_dir, num_features)

            test_metrics["test_classification"] = best_epoch

    if args.algorithm == 'dt':

        x_train, y_train = train_dataset.get_all_data()
        model            = tree.DecisionTreeRegressor().fit(x_train, y_train)
        y_train1         = model.predict(x_train)

        plot(np.array(y_train), y_train1, log_dir, "train")
        train_metrics = calc_metrics(y_train, y_train1, "train_")

        x_test, y_test = test_dataset.get_all_data()
        y_test1        = model.predict(x_test)

        plot(np.array(y_test), y_test1, log_dir, "test")
        test_metrics = calc_metrics(y_test, y_test1, "test_")

    if args.task == 'regression':
        write_summary(test_metrics, log_dir)

    return test_metrics


# ---------------------------- Learning Methods -----------------------------
def loocv(args, train_kwargs, test_kwargs):

    # Leave-one-out Cross-Validation (LOOCV)
    print ("------------- Leave-one-out Cross Validation Mode ---------------")

    # There needs to be multiple csv files to do cross-validation
    csv_files = glob.glob(os.path.join(args.data_dir, "**/*.csv"), recursive=True)

    df = list()
    cats = None
    bench2metrics = dict()

    for f in range(0, len(csv_files)):

        print (f"\n>>>>>>>>>>> Benchmarks No. {f+1}/{len(csv_files)} <<<<<<<<<<<")

        bench_name = os.path.splitext(os.path.basename(csv_files[f]))[0] 
        print (f">>>>>>>>>> {bench_name} <<<<<<<<<<\n") 

        bench_log_dir = os.path.join(args.log_dir, bench_name)
        os.makedirs(bench_log_dir, exist_ok=True)

        # cats is global category list which will be used as a reference
        if not cats:
            args.feature_to_class["path"] = os.path.join(args.data_dir, "cats.pkl")
            df = pd.concat((pd.read_csv(train_set, sep=",") 
                            for train_set in csv_files))
            _, cats = feature_to_class(df, **args.feature_to_class)
            args.feature_to_class["cat"] = cats
            args.num_classes = len(cats)

        loocv_temp = csv_files.copy()
        loocv_temp.pop(f)

        df = pd.concat((pd.read_csv(train_set, sep=",") 
                        for train_set in loocv_temp))
        df_test = pd.read_csv(csv_files[f], sep=",")

        # v3 computing the classes for both training and test
        df, _ = feature_to_class(df, **args.feature_to_class)

        df_test, _ = feature_to_class(df_test, **args.feature_to_class)

        train_data = os.path.join(bench_log_dir, 'train_data.csv')
        test_data  = os.path.join(bench_log_dir, 'test_data.csv')

        df.to_csv(train_data, index = False, sep=',') 
        df_test.to_csv(test_data, index = False, sep=',') 

        # Leaving f out of csv_files (f will be used for test only) 
        best_metrics = learn(args, train_data,
                             test_data,
                             bench_log_dir,
                             train_kwargs,
                             test_kwargs)

        bench2metrics[bench_name] = best_metrics

    # Write summary of all benchmarks
    print ("\n>>>>>>>>>>> Summary of all Benchmarks <<<<<<<<<<<")
    for bench, metrics in bench2metrics.items():
        print(f"Benchmark {bench}")
        print(metrics, "\n")


def standalone(df, args, train_kwargs, test_kwargs, portion:int=10):

    # Standalone Train/Test
    print ("------------------- Standalone Train/Test Mode: -----------------")

    # Number of rows for test in percentage
    test_rows = int(len(df.index) * portion/100)

    test  = df.iloc[0:test_rows, :]
    train = df.iloc[test_rows: , :]

    train = df.iloc[np.r_[0, test_rows + 1: np.shape(df)[0]], :]

    temp_dir = os.path.join(args.work_dir, 'temp/')
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)

    train_data = os.path.join(temp_dir, "train_data.csv")
    test_data  = os.path.join(temp_dir, "test_data.csv")

    print("Train-set:", np.shape(train),"Test-set:", np.shape(test))

    train.to_csv(train_data, index=False, sep=',')
    test.to_csv(test_data  , index=False, sep=',')

    best_metrics = learn(args, train_data,
                         test_data,
                         args.log_dir,
                         train_kwargs,
                         test_kwargs,
                        )

    write_summary(best_metrics, log_dir=args.log_dir, file_name="summary_classification.txt")


def kfold(df, args, train_kwargs, test_kwargs, k_folds:int=10):

    # Kfold Cross-Validation

    print ("------------- Kfold Cross Validation Mode -------------------------")
    # Define the K-fold Cross Validator
    k_folds = args.folds
    print(f"Number of folds: {k_folds}")
    kfold   = KFold(n_splits=k_folds, shuffle=True, random_state=2)

    for fold, (train_ids, test_ids) in enumerate(kfold.split(df)):

        print(f'FOLD {fold}')
        print('--------------------------------')
        train = df.iloc[train_ids]
        test  = df.iloc[test_ids]

        print(f"Train-set: {np.shape(train)}, Test-set: {np.shape(test)}")
        train_fold = "train_fold_" + str(fold) + ".csv"
        test_fold  = "test_fold_"  + str(fold) + ".csv"
        temp_dir = os.path.join(args.work_dir, 'temp/')
        if not os.path.exists(temp_dir):
            subprocess.run(f'mkdir -p {temp_dir}', shell = True)
        train_fold = os.path.join(temp_dir, train_fold)
        test_fold  = os.path.join(temp_dir, test_fold )

        train.to_csv(train_fold, index=False, sep=',')
        test.to_csv(  test_fold, index=False, sep=',')

        best_metrics = learn(args, train_fold,
                             test_fold,
                             args.log_dir,
                             train_kwargs,
                             test_kwargs,
                            )

        write_summary_classification(best_metrics, fold, log_dir=args.log_dir, file_name="summary_classification.txt")


def cv_method(df, args, train_kwargs, test_kwargs):
    """
    This function is used to select the learning algorithm
    """
    if args.cv == "loocv":
        return loocv(args, train_kwargs, test_kwargs)

    if args.cv == "standalone":
        return standalone(df, args, train_kwargs, test_kwargs)

    if args.cv == "kfold":
        return kfold(df, args, train_kwargs, test_kwargs)

    return "No algorithm is selected"

def check_params(args):
    if (args.task == 'classification') and (args.loss != 'cross_entropy') and (args.algorithm == 'nn'):
        print('task classification best use cross_entropy loss')
        return False
    
    if (args.task == 'regression') and (args.loss == 'cross_entropy') and (args.algorithm == 'nn'):
        print('cross_entropy loss is not suit for task regression')
        return False

    return True
# ------------------------------- The main body --------------------------------
def main():
    # updating train/test arguments
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs  = {'batch_size': args.test_batch_size}

    #check loss and task param
    if check_params(args) == False:
        return

    # setting required configuration for GPU
    if use_cuda:
        cuda_kwargs = {'num_workers': 0,
                       'pin_memory': True,
                       'shuffle': True,
                       }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # preprocessing steps
    #   1. creating working directories
    #   2. reading dataset file
    #   3. cleaning data
    create_log_folders(args)
    df = None
    if args.cv in {"standalone", "kfold"}:
        # "loocv" has its own data preparation
        df = data_loader(args)
        print(f"The data shape is {df.shape}. \n"
              f"The features are \n {df.columns.values}")

    #loocv data need additional processing
    if args.cv == 'loocv':
        prepare_loocv_data(args)

    # run learning algorithm (e.g., LOOCV, Standalone, Kfold)
    print(f"\nThe hyper-parameters are: \n"
          f"learning rate: {args.lr},\n"
          f"   batch size: {args.batch_size},\n"
          f"  # of epochs: {args.epochs} \n")

    cv_method(df, args, train_kwargs, test_kwargs)

    # clean up obsolete files from the work_path
    if args.erase_work:
        path = os.path.join(args.work_path, "temp/")
        os.system('rm -rf %s/*' % path)


if __name__ == '__main__':
    start_time = time.time()

    # preprocessing steps
    # 0. reading user_config file and update corresponding args
    # 1. set the root path
    args = sync_config(args.user_config)
    os.chdir(args.root_path)

    main()

    print(f"training time is {time.time() - start_time}")

# --------------------------------- The End! ----------------------------------
