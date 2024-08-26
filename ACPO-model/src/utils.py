import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import f1_score, accuracy_score, jaccard_score

plt.rcParams['agg.path.chunksize'] = 10000
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import onnx
    import onnx_tf.backend


def calc_classification_accuracy(label, output, log=False):
    """
    Parameters:
        label (s, 1) tensor: each row shows the label of corresponding sample
        output (C, s) tensor: each column corresponds to a sample. Length of each column is the number of classes.
    Note:
        each row of a column represents likelihood of data sample belongs to a specific class.
        For example, the value on (2, 3) of output shows the likelihood of sample 3 being in class 2.
    """
    pred     = output.max(1, keepdim=True)[1] 
    correct  = pred.eq(label.view_as(pred)).sum().item()
    total    = output.shape[0]
    accuracy = 100 * correct / total

    if log:
        print(f"Correct: {correct}, Total: {total}, Classification Accuracy: {accuracy}%")

    return correct, total, accuracy


def calc_metrics_classification(data, label, output, suffix=""):

    metrics = dict()
    metrics[suffix+"classification"] = calc_classification_accuracy(label, output)

    return metrics


def calc_metrics(actual, pred, suffix=""):
    metrics = dict()

    metrics[suffix+"mae"] = sklearn.metrics.mean_absolute_error(actual, pred)
    metrics[suffix+"mape"] = np.mean(np.abs((actual - pred) / actual))
    metrics[suffix+"mse"] = sklearn.metrics.mean_squared_error(actual, pred)

    return metrics


def plot(actual, pred, log_dir, name="plot"):
    plt.clf()
    ax = plt.gca()
    plt.grid()
    i = np.argsort(actual)
    plt.plot(actual[i], 'b-', label='Actual Data')
    plt.plot(pred[i], 'r-', label='Predictions', linewidth=0.5)
    #plt.plot(actual, pred, 'o')
    plt.legend(loc='lower right')
    plt.title(name)
    plt.xlabel("Test Data", fontweight='bold')
    plt.ylabel("Speedup", fontweight='bold')
    #ax.set_xticks(np.arange(0, len(pred), 200))
    ax.set_ylim(0.45, 1.8)
    plt.savefig(os.path.join(log_dir, name + ".png"), format='png', dpi=300)  


def plotloss(loss, log_dir, name="training-lossplot"):
    plt.clf()
    ax = plt.gca()
    plt.grid()
    plt.plot(loss, 'r-', label='MSE Loss', linewidth=0.5)
    #plt.plot(actual, pred, 'o')
    plt.legend(loc='lower right')
    plt.title(name)
    plt.xlabel("Training Batch", fontweight='bold')
    plt.ylabel("Loss", fontweight='bold')
    #ax.set_xticks(np.arange(0, len(pred), 200))
    ax.set_ylim(0, 1.5)
    plt.savefig(os.path.join(log_dir, name + ".png"), format='png', dpi=300)  


def plotgrad(grad, log_dir, name="training-gradplot"):
    plt.clf()
    ax = plt.gca()
    plt.grid()
    plt.plot(grad, 'r-', label='Grad', linewidth=0.5)
    #plt.plot(actual, pred, 'o')
    plt.legend(loc='lower right')
    plt.title(name)
    plt.xlabel("Training Batch", fontweight='bold')
    plt.ylabel("Gradients", fontweight='bold')
    #ax.set_xticks(np.arange(0, len(pred), 200))
    ax.set_ylim(0, 2.5)
    plt.savefig(os.path.join(log_dir, name + ".png"), format='png', dpi=300)  


def save_pb(torch_model, log_dir, num_features):
    # Export the trained model to ONNX
    inference_input = torch.randn(1, num_features).cuda()
    onnx_file = os.path.join(log_dir, "lu.onnx")
    torch.onnx.export(torch_model, inference_input, onnx_file)

    # Import the ONNX model to Tensorflow
    onnx_model = onnx.load(onnx_file)
    pb_file = os.path.join(log_dir, "lu.pb")
    tf_rep = onnx_tf.backend.prepare(onnx_model)
    tf_rep.export_graph(pb_file)


def is_better(new_metrics, best_metrics, metric="test_classification") -> bool:
    """This function returns True if new metric is better than to the current one.
    """
    if isinstance(new_metrics, dict):
        return (best_metrics is None) or (new_metrics[metric] > best_metrics[metric])

    return (best_metrics is None) or (new_metrics > best_metrics)


def write_summary(metrics, log_dir, file_name="summary.txt"):
    summary_file_path = os.path.join(log_dir, file_name)
    with open(summary_file_path, "w") as f:
        for key, value in metrics.items():
            f.write(key + ": " + str(value) + "\n")


def write_summary(metrics, log_dir, file_name="summary.txt"):
    """This function writes the summary of the classification into the file.
    """
    summary_file_path = os.path.join(log_dir, file_name)
    with open(summary_file_path, "w") as f:

        for key, value in metrics.items():
            # f.write(key + ": " + str(value) + "\n")
            line = "".join([key, ": ", str(value)])
            f.write(line + "\n")


def write_summary_classification(metrics, folds, log_dir, file_name="summary_classification.txt"):
    """This function appends classification summary of the current fold to the summary file.
    """
    summary_file_path = os.path.join(log_dir, file_name)
    with open(summary_file_path, 'a') as f:

        for key, value in metrics.items():
            # f.write("Fold " + str(folds) + " " + key + ": " + str(value) + "\n")
            line = "".join(["Fold ", str(folds), " ", key, ": ", str(value)])
            f.write(line + "\n")


def topK_labels(y_hat, k:int=1):
    """This function generates top k labels corresponding to the model's outputs.
    """
    k = min(k, y_hat.shape[1])

    if torch.is_tensor(y_hat):
        k_labels = torch.argsort(y_hat)[:, -k:]

    else:
        # sort, and counting predictions
        k_labels = np.argsort(y_hat)[:, -k:]

    return k_labels


def topk_metric(ds, y, y_hat, class_keys=None, top_n=None, filename=None, k:int=5, id=None, option:int=0):
    """This function is used to generate top5 accuracy metric for a given dataset and their prediction.

    Args:
        ds (pd.DataFrame): contains data points
        y (np.ndarray): contains correct label for each data point.
        y_hat (np.ndarray): prediction of the model on each data point.
        class_keys (pd.DataFrame, optional): dictionary map each class to an actual value. Defaults to None.
        top_n (list, optional): depicts top_n accuracy. Defaults to None.
        filename (str, optional): the csv file we want to attach the report. Defaults to None.
        k (int, optional): the number of top_k accuracy. Defaults to 5.
        id (list, optional): column index of ds we want to have in the csv file. Defaults to None.
        option (int, optional): determine the type of report. Defaults to 0.
            0: incorrect, report on miss-classified
            1: correct, report on classified correctly
            2: complete, a complete report.

    Returns:
        top_n (list): top_n accuracy for the given data points.
    """
    k = min(k, ds.shape[1])
    if top_n is None:
        top_n = [0] * k

    # sort, and counting predictions
    k_labels = topK_labels(y_hat, k)
    top_n += np.sum(k_labels == y[:,None], axis=0)[::-1]
    # top_n += np.sum(np.abs(k_labels - y[:,None])==0, axis=0)[::-1]

    if filename:
        if option == 0:
            idx = np.where(k_labels[:, -1] != y)[0]
        elif option == 1:
            idx = np.where(k_labels[:, -1] == y)[0]
        else:
            idx = np.array(range(len(k_labels)))

        pred       = class_keys.iloc[k_labels[idx, -1],: ]
        actual     = class_keys.iloc[y[idx],           : ]
        identifier = ds.iloc[idx, id]

        pred       = pred.set_index(pd.Index(range(len(idx))))
        actual     = actual.set_index(pd.Index(range(len(idx))))
        identifier = identifier.set_index(pd.Index(range(len(idx))))

        rows = pd.concat([identifier, pred, actual], axis=1, join="inner", ignore_index=True)

        # Writing to the end of the csv file
        with open(filename, 'a+') as csvfile:
            # csvwriter = csv.writer(csvfile)
            rows.to_csv(csvfile, mode='a', index=False, header=False)

    return top_n


def topK_accuracy(y, y_hat, k:int=1):
    """
    y_hat : (bs, n_labels)
    y : (bs,)
    """
    # y, y_hat should be tensor
    y = torch.from_numpy(y) if isinstance(y, np.ndarray) else y
    y_hat = torch.from_numpy(y_hat) if isinstance(y_hat, np.ndarray) else y_hat

    labels_dim = 1
    k = min(k, y_hat.size(labels_dim))
    # k_labels contains first k indices of high to low
    k_labels = torch.topk(input=y_hat, k=k, dim=labels_dim, largest=True, sorted=True).indices

    # True (1) if `expected label` in k_labels, False (0) if not
    a = ~torch.prod(input=torch.abs(y.unsqueeze(labels_dim) - k_labels), dim=labels_dim).to(torch.bool)

    # These two approaches are equivalent
    if False :
        y_pred = torch.empty_like(y)
        for i in range(y.size(0)):
            if a[i] :
                y_pred[i] = y[i]
            else :
                y_pred[i] = k_labels[i][0]
        #correct = a.to(torch.int8).numpy()
    else :
        a = a.to(torch.int8)
        y_pred = a * y + (1-a) * k_labels[:,0]
        #correct = a.numpy()

    f1 = f1_score(y, y_pred, average='weighted')*100
    #acc = sum(correct)/len(correct)*100
    acc = accuracy_score(y, y_pred)*100

    iou = jaccard_score(y, y_pred, average="weighted")*100

    return acc, f1, iou, y_pred


# TODO: this function will depreciated, after main.py runs with topk_metric
def top5_metric(ds, target, output, class_keys=None, top_n=None, filename=None):
    '''This function is used to generate top5 accuracy metric for a given dataset and their prediction
    '''
    if top_n is None:
        top_n = [0] * 5

    if torch.is_tensor(ds):
        top_5_results = torch.argsort(output)[:, -5:]
        top_n = [top_n[i-1] + torch.sum(top_5_results[:, [-i]] == target) for i in range(1, 6)]

        # # calculating metrics for evaluating the model
        # top_accuracy = [100 * top_n[i]/len(ds) for i in range(5)]

    else:
        # sort, and counting predictions
        top_5_results = np.argsort(output)[:, -5:]
        top_n = [np.sum(top_5_results[:, -i] == target, keepdims=True)
                + top_n[i-1]
                for i in range(1, 6)]

        # # calculating metrics for evaluating the model
        # top_accuracy = [100 * top_n[i]/len(ds.index) for i in range(5)]

    # for i in range(5):
    #     print(f"Top-{i+1} Predictions: {top_accuracy[i]}")

    if filename:
        miss_matches = np.where(top_5_results[:, -1] != target)[0]

        pred    = class_keys.iloc[top_5_results[miss_matches, -1],: ]
        actual  = class_keys.iloc[target[miss_matches],           : ]
        loop_IR = ds.iloc[miss_matches, 0].to_frame()

        pred    = pred.set_index(pd.Index(range(len(miss_matches))))
        actual  = actual.set_index(pd.Index(range(len(miss_matches))))
        loop_IR = loop_IR.set_index(pd.Index(range(len(miss_matches))))

        rows = pd.concat([loop_IR, pred, actual], axis=1, join="inner", ignore_index=True)

        # Writing to the end of the csv file
        with open(filename, 'a+') as csvfile:
            # csvwriter = csv.writer(csvfile)
            rows.to_csv(csvfile, mode='a', index=False, header=False)
            # csvwriter.writerow(rows)
            # csvwriter.writerow('')
            # for i in range(5):
            #     csvwriter.writerow(top_accuracy[i])

    return top_n
