import csv
import os
import pickle as pk
from itertools import product
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.decomposition import *
from sklearn.feature_selection import *
from sklearn.preprocessing import *
from torch.utils.data.dataset import Dataset
import random

import subprocess

def remove_duplicates(data, keys:list=None, order_key:str=None, ascending:bool=True, keep:str="last"):
    """This function removes duplicates from the given dataset based on the given keys.

    Example:
        order_key is not None -> sort data by order_key -> keep first or last (same as max).
        order_key is None -> unsorted data -> keep first or last.

    Args:
        data (pd.DataFrame): data frame with column vectors as features and rows are samples.
        keys (list, optional): keys to check data uniqueness. Defaults to None. If key is not given, the whole features become a key.
        keep (str, optional): it keep first, last, or max instance on an specific attribute. Defaults to "last".
        order_key (str, optional): the function keeps the max based on the given order_key. Defaults to None.
    """
    if not keys or keys == "all":
        keys = data.columns

    before = len(data.index)
    if order_key:
        data.sort_values(by=order_key, ascending=ascending, inplace=True)
        keep = "last" if keep == "max" else keep

    data.drop_duplicates(subset=keys, inplace=True, keep=keep)
    data.sort_index(inplace=True)

    after = len(data.index)

    if before == after:
        print(f"No duplicates found in the dataset.")
    else:
        print(f"By removing {before - after} duplicates, from {before}, we get {after} data points.")


def get_index(data, attr:str, opr:str, value):
    """This function returns index of data points who satisfies the condition.

    Args:
        data (pd.DataFrame): the data frame to search.
        attr (str): specific feature / dimension on data.
        opr (str): operator used for comparison.
        value (str, float, int): the value to compare against the data.
    """
    idx = None

    if opr == "<":
        idx = data[data[attr] < value].index
    elif opr == ">":
        idx = data[data[attr] > value].index
    elif opr in {"=", "=="}:
        idx = data[data[attr] == value].index
    elif opr in {"!", "~"}:
        idx = data[data[attr] != value].index
    elif opr == ">=":
        idx = data[data[attr] >= value].index
    elif opr == "<=":
        idx = data[data[attr] <= value].index

    return idx


def prune_data(data, key:str=None, values:list=None, queries:List=None, logic:str="AND"):
    """
    This function removes data points that has specific value on a feature.
    There are two layers filtering:
        - First filter occurs using key, and values.
        - Second filter occurs using queries on top of the first filter.
    If the first filter is not available, then the second filter applies on all data.

    Example 1:
        key    = "BenchmarkName"
        values = ["consumer_jpeg_c",
                  "automotive_susan_e",
                  "consumer_lame",
                  "automotive_susan_c",
                 ]

        as a result, whatever rows (data points) has one of these values on its
        "BenchmarkName" key (i.e., feature) will be removed.

    Example 2:
        key    = "BenchmarkName"
        values = ["consumer_jpeg_c",
                  "automotive_susan_e",
                 ]
        queries = ["score > 5.2"]
        So, the function will drop data points with values on "Benchmark"
        where their score is higher than 5.2.

    Example 3:
        key    = "BenchmarkName"
        values = "consumer_jpeg_c"
        queries = ["score > 5.2"]

    Example 4:
        queries = ["score > 5.2"]

    Note that, if the queries is true, the function will drop that data point.

    Args:
        data (pd.DataFrame): the data frame to be pruned.
        key (str): the key of the feature to be pruned.
        values (list): the list of values to be pruned.
        queries (str, Optional): attribute, operator, value. Defaults to None.
                          operators: {>, <, =, !, >=, <=}
        logic (str, Optional):
            if AND/Intersection is selected: all rows satisfying all conditions.
            if OR/Union is selected: all rows satisfying either conditions.
    """
    if not values and not queries:
        print("There is not values nor queries to prune dataset.")
        return

    before = len(data.index)

    idx = data.index
    if key and values:
        values = [values] if isinstance(values, str) else values

        idx = pd.Index([])
        # unconditionally remove rows with specific values on column key
        for value in values:
            idx = idx.union(data[data[key] == value].index)

    if logic in {"AND", "Intersection"}:
        idx_tmp = data.index
    else:
        idx_tmp = pd.Index([])

    queries = queries if queries else list()
    for q in queries:
        if logic in {"AND", "Intersection"}:
            # idx_tmp &= data.query(q).index
            idx_tmp = idx_tmp.intersection(data.query(q).index)
        else:
            # idx_tmp |= data.query(q).index
            idx_tmp = idx_tmp.union(data.query(q).index)

    idx &= idx_tmp if queries else idx

    data.drop(idx, inplace=True)
    after = len(data.index)

    if before == after:
        print("There is not any row found to satisfies the query.")
    else:
        print(f"By pruning, {before - after} data points are filtered out, from {before}, we have {after} data points.")


def drop_feature(data, keys:list):
    """
    This function drops all the features listed in the keys from the given data set.
    """
    if not keys:
        return

    feature_set = set(data.columns)
    before = len(data.columns)
    for feature in keys:
        if feature in feature_set:
            data.drop(columns=feature, axis=1, inplace=True)
        else:
            print(f"The feature '{feature}' is not in the feature_set.")

    after = len(data.columns)

    if after == before:
        print("Number of features remains the same. Nothing dropped.")
    else:
        print(f"By dropping {before - after} features, from {before}, we keep {after} features.")


def add_feature(data, key:str, value, loc:int=None):

    if loc:
        data.insert(loc=loc, column=key, value=value, allow_duplicates=True)
    else:
        data[key] = value


def find_constant_feature(data, key:List[str]=None, debug:bool=False, alphanum:bool=False):
    """This function returns all features who do not add any value in the model.
    For example, if all values are the same (i.e., min=max) or std=0.
    Such features do not differentiate classes and thus not helpful.

    Args:
        data (pd.DataFrame): dataset where each column represents a feature.
        key (List[str], Optional): specific features to investigate. Default to None.
        alphanum (bool, optional): whether to consider features with non-numeric values as well. Default is False.

    returns:
        constant_features (List(str)): list of features which does not add any value in the model.
    """

    constant_feature = list()
    if not key:
        key = data.columns

    for column in key:

        if len(data[column].unique()) > 1:
            continue

        if not alphanum and (data[column].dtypes == "object"):
            continue

        constant_feature.append(column)

        # if (data[column].min() == data[column].max() or data[column].std() == 0):
        #     constant_feature.append(column)

    print(
        f"The number of features that do not change in value is {len(constant_feature)}."
    )
    if debug:
        print(f"The constant features are {constant_feature}")

    return constant_feature


def remove_constant_feature(data, debug=False):

    constant_features = find_constant_feature(data, debug=debug)
    drop_feature(data, constant_features)


def find_null(data, debug=False):
    """This function check the data set for null values and return index of them.

    Args:
        data (pd.DataFrame): data points to check for null values.

    Returns:
        idx (list): index of rows which contain null values.
    """
    idx = data[data.isnull().any(axis=1)].index.to_list()

    if not len(idx):
        print("There is not any data point with NA value.")

    else:
        print(f"The number of rows which contains null value(s) is: {len(idx)}")
        if debug:
            print(f"These rows in data set contain Null values: {idx}")

    return idx


def remove_null(data, debug=False):

    idx = find_null(data, debug=debug)

    before = len(data)

    data.drop(idx, inplace=True)

    after = len(data)

    if before != after:
        print(f"By removing null data, data points reduced from {before}, to {after}.")


def head_count(data, subset, decimals:int=3, **params):
    """_summary_

    Args:
        data (pd.DataFrame): dataset
        subset (str, list): the column which counting values
        decimals (int, optional): the decimals place for the floating point values.
        normalize (bool): whether to normalize
        sorted (bool): whether to sort
        ascending (bool): whether to sort ascending or descending
        dropna (bool): whether to drop NA values

    Returns:
        pd.DataFrame: a DataFrame containing headcount and their percentage values.
    """

    df = data.value_counts(subset=subset, **params).to_frame("count")
    df["%"] = (100 * df["count"] / len(data)).round(decimals=decimals)

    return df


def feature_to_class(data, keys:List[str], cat:dict=None, path:str=None):
    """This function computes categories as combination of given keys and revise the data adding classes column to it.

    Args:
        data (pd.DataFrame): dataset
        keys (List[str]): _description_
        cat (dict, optional): combination of keys to classes mappings. Defaults to None.
        path (str, optional): the path to save category mappings. Defaults to None.

    Returns:
        DataFrame: A DataFrame by adding classes column to the given data considering the categories.
        dict: A dictionary mapping combination of keys into classes/categories.
    """
    if not cat:
        if len(keys) > 1:
            cat = list(product(*(data[key].sort_values().unique() for key in keys)))
        else:
            cat = list(data[keys[0]].sort_values().unique())

        if path:
            pk.dump(cat, open(path, "wb"))

    num_of_classes = len(cat)
    out = pd.DataFrame(cat, columns=keys).assign(Classes=range(num_of_classes))
    if "Classes" not in data.columns:
        data = data.merge(out, on=keys, how="left")

    unique_num_of_classes = len(data["Classes"].unique())

    print(f"The number of possible classes: {num_of_classes}")
    print(f"The unique number of classes: {unique_num_of_classes}")
    print(*keys, sep=", ", end=" -> category mapping is: \n")
    print(data.set_index(keys)["Classes"].to_dict())
    print("Head count and percentage of classes are:")
    print(head_count(data, subset="Classes", sort=False))

    return data, cat


def plot_matrix(data, log_dir:str=None):

    # plot correlation and save the plot
    sns.set(font_scale=0.4)
    # plt = sns.heatmap(df.corr(), annot=True, annot_kws={'size': 8}).get_figure()
    plt = sns.heatmap(data, yticklabels=1, xticklabels=1, vmin=0, vmax=1).get_figure()

    if log_dir:
        plot_path = os.path.join(log_dir, "corr_plot_" + ".png")
        plt.savefig(plot_path, format="png", dpi=600)

def features_relation(data, key:str=None, correlation:bool=True, plot=False, log_dir:str=None, sort=False, ascending=False, threshold:float=0.0):
    """This function returns correlation between features.

    Args:
        data (DataFrame): dataset; columns are features and rows are data points
        correlation (bool, optional): whether to calculate correlation if True or covariance if False. Defaults to True.
        key (str, optional): the feature that correlation (or covariance) measure . Defaults to None.
        sort (bool, optional): whether to sort features or not. Defaults to False.
        ascending (bool, optional): whether to sort features ascending. Defaults to False.
        threshold (float, optional): determines the acceptable level of correlation (or covariance).
        log_dir (string, optional): if we want to save the correlation (or covariance) results to a log file.
        plot (bool, optional): whether to plot and save the correlation (or covariance) results. Defaults to False.

    Returns:
        If the key is provided, it returns the correlation (or covariance) of (features, key).
        If the key is not provided, it returns the correlation (or covariance) of (features, Classes).
        If key is None, it returns the correlation (or covariance) between any pair of features.
    """
    if correlation:
        df = data.corr()
        file_name = "correlationFTs"
        metric = "P.Coef"
    else:
        # covariance
        df = data.cov()
        file_name = "covarianceFTs"
        metric = "cov"

    if log_dir:
        file_path = os.path.join(log_dir, file_name + ".csv")
        df.to_csv(file_path, index=False, sep=",")

    # plot correlation and save the plot
    if plot:
        plot_matrix(df, log_dir=log_dir)

    # correlation of each pair of features
    if not key:
        return df

    # In case key is a list, it means we already added "Classes" attribute to df.
    key = key if isinstance(key, str) else "Classes"

    # correlation of each feature with specific key or "Classes", sorted or unsorted.
    if sort:
        df = df[key].sort_values(key=abs, ascending=ascending).to_frame()
    else:
        df = df[key].to_frame()

    if threshold:
        df[f"|{metric}|>{threshold *100}%"] = df[key].apply(lambda x: f"{'Good' if (abs(x) > threshold) else 'Weak'}")

    return df


def create_log_folders(args) -> None:

    print("--- Making Folders ---")

    # making folder on the given path if required
    os.makedirs(args.work_path, exist_ok=True)
    os.makedirs(os.path.join(args.work_path, "temp/"), exist_ok=True)
    os.makedirs(os.path.join(args.work_path, "test/"), exist_ok=True)

    # create log directory / trained model
    args.log_dir = os.path.join(args.log_path, args.model["name"])
    os.makedirs(args.log_dir, exist_ok=True)

def shuffle_raw_data(raw_data, shuffled_data) -> None:
    print("--- Shuffle Raw Data ---")

    with open(raw_data, 'r') as raw_data_file:
        reader = csv.reader(raw_data_file)
        content = list(reader)
        header = content[0]
        data = content[1 : ]
    random.shuffle(data)

    with open(shuffled_data, 'w') as shuffled_data_file:
        writer = csv.writer(shuffled_data_file)
        writer.writerow(header)
        writer.writerows(data)

def prepare_loocv_data(args):
    # TODO: Split training data
    # temporary processing for loocv.
    # It will be fixed as soon as possible
    raw_data = os.path.join(args.data["path"], args.data["file_name"] + ".csv")
    loocv_data_path = args.data_dir

    print("--- loocv data parepare under development ---")
    if not os.path.exists(args.data_dir):
        os.mkdir(args.data_dir)
    subprocess.run('cp ./work/temp/*fold*.csv ./data', shell = True)
    pass

def data_loader(args):

    if not args.data["file_name"]:
        raise TypeError("data file expected!")

    # FIXME: should consistent with args data flow
    raw_data = os.path.join(args.data["path"], args.data["file_name"] + ".csv")
    shuffled_data = os.path.join(args.work_path, "temp/", "shuffled" + ".csv")
    shuffle_raw_data(raw_data, shuffled_data)

    print("--- Reading Data File ---")
    # preprocessing step: data preparation, data cleaning

    if args.build_data:
        # data preparing and preprocessing
        df = pd.read_csv(raw_data, sep=",")

        # Computing & save global category dictionary given the whole data
        args.feature_to_class["path"] = os.path.join(args.data_dir, "cats.pkl")
        df, _ = feature_to_class(data=df, **args.feature_to_class)

        # Shuffle rows before divide into train/test
        df = df.sample(frac=1).reset_index(drop=True)
        df.to_csv(shuffled_data, index=False, sep=",")
    else:
        df = pd.read_csv(shuffled_data, sep=",")

    # update number of classes
    args.num_classes = len(np.unique(df[args.class_key].values))

    # ------------- cleaning data -------------
    if args.remove_duplicates['keys']:
        remove_duplicates(df)
    # remove_duplicates(df, **args.remove_constant_feature)

    if args.prune_data['key']:
        prune_data(df, **args.prune_data)

    # remove null entries
    remove_null(df)

    # remove constant features
    remove_constant_feature(df, **args.remove_constant_feature)

    # dropping some features
    if args.drop_feature['keys']:
        drop_feature(df, **args.drop_feature)

    # stat of data
    print(f"The dataset has {df.shape} rows and columns.\n")
    print(features_relation(df, **args.features_relation))

    return df

class CSVDataset(Dataset):
    # Use x_col_start with ['IR','Function','Loop']  and
    # ends with ['UP.Count','UnrollType','Classes']
    # y_col is the label/class -> ['Classes']
    def __init__(
        self,
        file_name,
        incl_noinline_features=False,
        feature_scale=True,
        feature_transform_pca=True,
        feature_select=True,
        x_col_start=0,
        x_col_end=-2,
        y_col=-1,
        log_dir=None,
        mode="train",
        debug=[],
    ):

        # validation checks
        assert x_col_end < 0
        assert y_col < 0

        self.x_col_start = x_col_start
        self.x_col_end = x_col_end
        self.y_col = y_col
        self.incl_noinline_features = incl_noinline_features
        self.mode = mode
        self.log_dir = log_dir
        self.dump_dir = (
            os.path.join(self.log_dir, "data_preprocess.dbg") if self.log_dir else None
        )
        self.dump_count = 0
        self.debug = debug

        # read csv file and load row data into variables
        df = pd.read_csv(file_name, sep=",")
        # plot speedups with respect to -O3
        # self.plot_global_speedup(self, df)

        df_base = None

        # Step 1/6
        # Removing Duplicates
        print("--- Removing Duplicate Rows ---")
        print(f"Size before preprocessing: {df.shape}")

        # Step 3/6
        # Pruning Uncorrelated Features using
        # heatmap() with df.corr() and df.corr().index
        if self.mode == "train":
            features_relation(df, log_dir=self.log_dir, plot=True)

        print("\n--- Pruning Uncorrelated Features ---")
        # v2 only
        # uncorr_ft = ['UP.Partial', 'UP.Runtime', 'UP.UpperBound']
        uncorr_ft = []
        df.drop(uncorr_ft, axis=1, inplace=True)

        print(f"After removing uncorrelated features: {df.shape}")
        if "drop-uncorrelated-features" in self.debug:
            self.dump(df, "after_drop_uncorrelated_features")

        # Step 4/6
        # Feature Scaling
        df, df_base = self.__scale_features(df, df_base, feature_scale)

        # Step 5/6
        # PCA
        df, df_base = self.__pca_features(df, df_base, feature_transform_pca)
        print(
            f"Training data after returning from PCA: \n"
            f"(Data, Features): {df.shape} \n"
        )

        # Step 6/6
        # Feature Selection
        df, df_base = self.__select_features(df, df_base, feature_select)
        print(
            f"Training data after returning from feature select: \n"
            f"(Data, Features): {df.shape} \n"
        )

        self.df, self.df_base = df, df_base
        # print(f"And here are the features: \n{self.df.columns[self.x_col_start:self.x_col_end]} \n")

        # Assuming: first 3 columns are non-feature details,
        # last 4 columns are 'UP.Count,UnrollType,Hash,Classes'
        print(
            f"Final training data after preprocessing: \n"
            f"(Data, Features): {df.shape} \n"
            f"Classes: {self.num_classes()} \n"
        )

        if "final" in self.debug:
            self.dump(df, "final")

    def __len__(self):
        return len(self.df.index)

    def num_features(self):
        num_features = len(self.df.columns) - self.x_col_start + self.x_col_end

        if self.incl_noinline_features:
            num_features *= 2

        return num_features

    def num_classes(self):
        return len(self.df["Classes"].unique())

    def get_all_data(self):
        if self.incl_noinline_features:
            self.df = pd.merge(
                self.df,
                self.df_base,
                on=["benchmark", "function", "program_input"],
                how="outer",
            )
            self.df.drop(
                ["iteration_y", "callsites_y", "FunctionSpeedup_y", "GlobalSpeedup_y"],
                axis=1,
                inplace=True,
            )

            # move FunctionSpeedup_x and GlobalSpeedup_x to the end
            cols = self.df.columns.tolist()
            cols.remove("FunctionSpeedup_x")
            cols.remove("GlobalSpeedup_x")
            cols.extend(["FunctionSpeedup_x", "GlobalSpeedup_x"])
            self.df = self.df[cols]

            self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
            self.df.dropna(inplace=True)

        x = self.df.iloc[:, self.x_col_start : self.x_col_end].values.astype(float)
        y = self.df.iloc[:, self.y_col]

        return x, y

    def __getitem__(self, idx):
        x = self.df.iloc[idx, self.x_col_start : self.x_col_end].values.astype(float)
        y = self.df.iloc[idx, self.y_col]

        if self.incl_noinline_features:
            # for each row, append features of a caller with
            # some inlining to a caller with no inlining
            function = self.df.iloc[idx]["function"]
            benchmark = self.df.iloc[idx]["benchmark"]
            no_inlining_row = self.df_base.loc[self.df_base["function"] == function]
            no_inlining_row = no_inlining_row[no_inlining_row["benchmark"] == benchmark]
            x_no_inlining = pd.to_numeric(
                no_inlining_row.iloc[0, self.x_col_start : self.x_col_end]
            )

            # FIXME: temporary hack to handle cases where we can't find the 'no-inlined' version of a function
            if x_no_inlining.shape[0] == 0:
                raise Exception(
                    "ERROR: can't find no-inlining-features of function: ",
                    function,
                    " in benchamrk: ",
                    benchmark,
                )

            x_no_inlining = np.squeeze(x_no_inlining)
            # FIXME: a hack to deal with multiple rows with different input_names (maybe take the average?)
            x = np.concatenate((x, x_no_inlining))

        # Defining the target/label column for Torch dataloader (Regression/Classification)
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor([y], dtype=torch.long)
        # Use torch.float32 for regression tasks
        # y_tensor = torch.tensor([y], dtype=torch.long)
        # print("Data shape:", np.shape(x_tensor.detach().cpu().numpy()), "Label shape: ", np.shape(y_tensor.detach().cpu().numpy()),"\n")
        return x_tensor, y_tensor

    def get_feature_scaler(self):
        return self.sc

    def get_feature_pca(self):
        return self.pca

    def get_feature_selector(self):
        return self.sel

    def __scale_features(self, df, df_base, feature_scale):
        df_X = df.iloc[:, self.x_col_start : self.x_col_end]
        df_Y = df.iloc[:, self.y_col]

        self.sc = None
        if feature_scale is True:
            self.sc = StandardScaler()
            self.sc.fit(df_X)
            # We need this scaling file at inference
            # print("Scaling for: ", self.mode, df_X)
            if self.mode == "train":
                pk.dump(self.sc, open(os.path.join(self.log_dir, "sc.pkl"), "wb"))
            elif self.mode == "test":
                pk.dump(self.sc, open(os.path.join(self.log_dir, "sc_test.pkl"), "wb"))
        elif feature_scale is False:
            self.sc = None
        else:
            self.sc = feature_scale

        if self.sc is not None:
            df.iloc[:, self.x_col_start : self.x_col_end] = self.sc.transform(
                df.iloc[:, self.x_col_start : self.x_col_end]
            )
            if df_base is not None:
                df_base.iloc[:, self.x_col_start : self.x_col_end] = self.sc.transform(
                    df_base.iloc[:, self.x_col_start : self.x_col_end]
                )

            if "feature-scaling" in self.debug:
                self.dump(df, "after_feature_scaling")

        return df, df_base

    def __pca_features(self, df, df_base, feature_transform_pca):
        df.reset_index(drop=True, inplace=True)
        df_X = df.iloc[:, self.x_col_start : self.x_col_end]
        df_Y = df.iloc[:, self.y_col]

        self.pca = None
        if feature_transform_pca is True:
            self.pca = PCA(n_components=15)
            self.pca.fit(df_X)
            # We need this scaling pca file at inference
            pk.dump(self.pca, open(os.path.join(self.log_dir, "pca.pkl"), "wb"))
        else:
            self.pca = feature_transform_pca

        if self.pca:
            df_pca = pd.DataFrame(self.pca.transform(df_X))
            df_pca = pd.concat([df[df.columns[0 : self.x_col_start]], df_pca], axis=1)
            df_pca = pd.concat([df_pca, df[df.columns[self.x_col_end :]]], axis=1)

            df = df_pca.copy()
            if "pca" in self.debug:
                self.dump(df, "after_pca")

        return df, df_base

    def __select_features(self, df, df_base, feature_select):
        df_X = df.iloc[:, self.x_col_start : self.x_col_end]
        df_Y = df.iloc[:, self.y_col]

        if feature_select is True:
            # This method for feature selection might remove all columns
            # if used after PCA
            self.sel = VarianceThreshold(threshold=(0.01))
            self.sel.fit(df_X)

            # self.sel = SelectKBest(chi2, k=100)
            # self.sel.fit(df_X, df_Y)

            # lsvc = LinearSVC(C=0.01, penalty="l1", dual=False)\
            # .fit(df_X, df_Y)
            # self.sel = SelectFromModel(lsvc, prefit=True)

            # clf = ExtraTreesClassifier(n_estimators=50)
            # self.sel = clf.fit(df_X, df_Y)
        elif feature_select is False:
            self.sel = None
        else:
            self.sel = feature_select

        if self.sel is not None:
            print("\n--- Select Features ---")
            df = df.drop(
                df.columns[self.x_col_start + self.sel.get_support(indices=True)],
                axis=1,
            )
            if df_base is not None:
                df_base = df_base.drop(
                    df_base.columns[
                        self.x_col_start + self.sel.get_support(indices=True)
                    ],
                    axis=1,
                )

            if "feature-select" in self.debug:
                self.dump(df, "after_feature_select")

        return df, df_base

    def dump(self, df, name):
        if self.dump_dir:
            if os.path.exists(self.dump_dir) is False:
                os.makedirs(self.dump_dir)

            # dump csv file
            if "csv" in self.debug:
                df.to_csv(
                    os.path.join(
                        self.dump_dir,
                        self.mode + "_" + str(self.dump_count) + "_" + name + ".csv",
                    ),
                    index=False,
                    sep=",",
                )

            # plot correlation map
            if "heatmap" in self.debug:
                plt = sns.heatmap(df.corr()).get_figure()
                plt.savefig(
                    os.path.join(
                        self.dump_dir,
                        self.mode
                        + "_"
                        + str(self.dump_count)
                        + "_"
                        + name
                        + "_corr_plot_"
                        ".png",
                    ),
                    format="png",
                    dpi=300,
                )

            # plot pair plots
            # TODO: Commented it for now because it is very time consuming
            if "pairplot" in self.debug:
                plt = sns.pairplot(df, hue="speedup", height=2.5)
                plt.savefig(
                    os.path.join(
                        self.dump_dir,
                        self.mode
                        + "_"
                        + str(self.dump_count)
                        + "_"
                        + name
                        + "_pair_plot_"
                        ".png",
                    ),
                    format="png",
                    dpi=300,
                )

            self.dump_count += 1

    def plot_speedup_corr(self, df):
        plots_dir = os.path.join(self.dump_dir, "corrplots")
        for benchmark in df["benchmark"].unique():
            os.makedirs(os.path.join(plots_dir, benchmark), exist_ok=True)
            df_bench = df[df["benchmark"] == benchmark]

            for idx, caller in enumerate(df_bench["function"].unique()):
                df_caller = df_bench[df_bench["function"] == caller]

                fig, axs = plt.subplots(2, 1)
                fig.suptitle(caller)

                axs[0].scatter(df_caller["FunctionSpeedup"], df_caller["GlobalSpeedup"])
                axs[0].set_xlabel("FunctionSpeedup")
                axs[0].set_ylabel("GlobalSpeedup")

                axs[1].scatter(
                    df_caller["avg_execution_time"], df_caller["global_time"]
                )
                axs[1].set_xlabel("avg_execution_time")
                axs[1].set_ylabel("global_time")

                plt.tight_layout()

                fig.savefig(os.path.join(plots_dir, benchmark, caller + ".png"))
                plt.close()

    # print speedup with respect to -O3
    def plot_global_speedups(self, df):
        # TODO: @Frank include -O3 in the spreadsheet so we won't need this hack
        bench2base = {
            "400.perlbench": 40073848973,
            "401.bzip2": 39767486110,
            "429.mcf": 76902940747.3333,
            "458.sjeng": 1103971535052,
        }
        for benchmark in df["benchmark"].unique():
            plt.clf()
            ax = plt.gca()
            plt.grid()

            df_benchmark = df.loc[df["benchmark"] == benchmark]
            df_benchmark["GlobalSpeedup"] = df["global_time"] / bench2base[benchmark]
            # df_benchmark = df_benchmark.drop(df_benchmark[df_benchmark['GlobalSpeedup'].eq(0)].index)
            plt.plot(
                df_benchmark["GlobalSpeedup"].unique(), "b*", label="Program Speedup"
            )
            plt.title(benchmark)
            plt.xlabel("Configuration", fontweight="bold")
            plt.ylabel("Speedup", fontweight="bold")
            plt.savefig(
                os.path.join(self.log_dir, benchmark + "_speedup.png"),
                format="png",
                dpi=300,
            )


class CSVLogger:
    def __init__(self, file_name):
        self.file_name = file_name
        self.has_header = False

    def append(self, epoch:int, train_log:float, test_log:float):

        if self.has_header is False:

            with open(os.path.join(self.file_name), "w") as f:
                csv_log = csv.writer(f)
                csv_log.writerow(
                    ["epoch", "train classification", "test classification"]
                )
                self.has_header = True

        with open(os.path.join(self.file_name), "a") as f:
            csv_log = csv.writer(f)
            csv_log.writerow([epoch, train_log, test_log])

    def append_df(self, epoch:int, batch_log):

        if not isinstance(batch_log, pd.DataFrame):
            return "you should use append method instead!"

        if self.has_header is False:
            with open(os.path.join(self.file_name), "w") as f:
                csv_log = csv.writer(f)
                csv_log.writerow(["batch", "epoch", "correct", "total", "accuracy"])
                self.has_header = True

        batch_log.insert(0, "epoch", epoch)
        batch_log.to_csv(self.file_name, mode="a", index=True, header=False)
