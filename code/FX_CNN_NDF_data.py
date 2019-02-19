# Script: FX_CNN_NDF_data
# Author: Tobias Broeckl
# Date: 27.01.2019

# Script to further prepare the preprocessed data for training.
# Includes transformation to 2D structure, normalization, label generation, batching and dataset splitting.
# prep_data() is the main function of this script, it is called by the FX_CNN_NDF_train script.
# For more details please see the corresponding jupyter notebook script.

import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset

def gen_mat(daily_forecast, currency):

    if daily_forecast:
        filename = currency + "_prep_day.csv"
        data = pd.read_csv(filename, sep=",")
        data_mat = data.loc[:, ["hour", "delta_days", "close"]]
        data_mat.set_index(['hour', 'delta_days'], inplace=True, drop=True)
        data_mat = data_mat.unstack(level=1)
        data_mat.fillna(method='bfill', inplace=True)
        data_mat.fillna(method='ffill', inplace=True)
        data_mat = data_mat.stack()
        data_mat = data_mat.unstack(level=0)
        data_mat = pd.DataFrame(data_mat.values)

    else:
        filename = currency + "_prep_hour.csv"
        data = pd.read_csv(filename, sep=",")
        data_mat = data.loc[:, ["date", "hour", "minute", "close"]]
        data_mat.set_index(['date', 'hour', 'minute'], inplace=True, drop=True)
        data_mat = data_mat.unstack(level=2)
        data_mat = data_mat.reset_index(drop=True)
        data_mat = data_mat.T
        data_mat.fillna(method='bfill', inplace=True)
        data_mat.fillna(method='ffill', inplace=True)
        data_mat = data_mat.T

    return data_mat

def gen_delta(data_mat, method, periods_ahead):

    # Method 1:
    if method == 1:
        p = data_mat.median(axis=1)
        p = p.pct_change(periods_ahead)
        p = p.shift(periods_ahead * -1)
        p = p.iloc[:-periods_ahead]

    # Method 2:
    elif method == 2:
        medians = data_mat.median(axis=1)
        p = []

        for i in range(len(medians) - periods_ahead):
            temp =(data_mat.loc[(i + 1):(i + periods_ahead), :] / medians[i]) - 1
            p.append(np.median(temp.values))

        p = pd.Series(p)

    else:
        raise NotImplementedError

    return p

def classify(n_class, p):

    if n_class == 2:
        bins = [-1, 0, 1]
        labels = [0, 1]

    elif n_class == 4:
        p25 = np.percentile(p, 25)
        p75 = np.percentile(p, 75)
        mean = np.mean([-p25, p75])
        bins = [-1, round(-mean, 4), 0, round(mean, 4), 1]
        labels = [0, 1, 2, 3]

    else:
        raise NotImplementedError

    classes = pd.cut(p, bins=bins, include_lowest=True, labels=labels)

    return classes

def batching(daily_forecast, periods_ahead, data_mat, y_classes):

    # Batching
    y_batches = []
    x_batches = []

    if daily_forecast:
        x_batch_length = 24
    else:
        x_batch_length = 60

    for i in range(data_mat.shape[0] - x_batch_length + 1 - periods_ahead):
        y_batch = np.asarray(y_classes.iloc[i + x_batch_length - 1])
        x_batch = np.asarray(data_mat.iloc[i:i + x_batch_length, :])
        y_batches.append(y_batch)
        x_batches.append(x_batch)

    return x_batches, y_batches

def min_max_norm(x_batches):

    x_batches_norm = []

    for i in range(len(x_batches)):
        min = x_batches[i].min()
        max = x_batches[i].max()
        x_batch_norm =(x_batches[i] - min) /(max - min)
        x_batches_norm.append(x_batch_norm)

    return x_batches_norm

def df_to_tensor(daily_forecast, x_batches_norm, y_batches):

    if daily_forecast:
        x_batches = torch.from_numpy(np.asarray(x_batches_norm)).type(torch.FloatTensor).view(-1, 24, 24)
    else:
        x_batches = torch.from_numpy(np.asarray(x_batches_norm)).type(torch.FloatTensor).view(-1, 60, 60)
    y_batches = torch.from_numpy(np.asarray(y_batches)).type(torch.IntTensor).view(-1, 1, 1)

    return x_batches, y_batches

def data_split(x_batches, y_batches, type ,split,k,img_count):

    if type == 1: # random
        np.random.seed(708145)
        rand = np.random.choice(2, img_count, p = [1 - split, split])
        training_idx = np.argwhere(rand == 1)
        testing_idx = np.argwhere(rand == 0)

    if type == 2: # chronologic
        end_trainset = int(img_count*split)
        data_length = np.arange(img_count)
        training_idx = np.argwhere(data_length <= end_trainset)
        testing_idx = np.argwhere(data_length > end_trainset)

    if type == 3: # test_only
        end_trainset = k
        data_length = np.arange(img_count)
        training_idx = np.argwhere(data_length <= end_trainset)
        testing_idx = np.argwhere(data_length > end_trainset)

    x_trainset = x_batches[training_idx, :, :]
    y_trainset = y_batches[training_idx, :, :]

    x_testset = x_batches[testing_idx, :, :]
    y_testset = y_batches[testing_idx, :, :]

    full_trainset = TensorDataset(x_trainset, y_trainset)
    full_testset = TensorDataset(x_testset, y_testset)

    return x_trainset, y_trainset, full_trainset, full_testset

def cross_val(x_trainset, y_trainset, k):
    training_sets_cv = []
    validation_sets_cv = []

    kf = KFold(n_splits=k)  # k = number of folds(min = 2)

    for train_idx, val_idx in kf.split(x_trainset):
        x_val = x_trainset[val_idx, :]
        y_val = y_trainset[val_idx, :]

        x_train = x_trainset[train_idx, :]
        y_train = y_trainset[train_idx, :]

        training_final = TensorDataset(x_train, y_train)
        validation_final = TensorDataset(x_val, y_val)

        training_sets_cv.append(training_final)
        validation_sets_cv.append(validation_final)

    return training_sets_cv, validation_sets_cv

def prep_data(daily_forecast, periods_ahead, n_class, split, class_method, k, currency):
    os.chdir("../data")
    data_mat = gen_mat(daily_forecast, currency)

    delta = gen_delta(data_mat, class_method, periods_ahead)
    labels = classify(n_class, delta)

    non_norm_mat, labels = batching(daily_forecast, periods_ahead, data_mat, labels)
    norm_mat = min_max_norm(non_norm_mat)
    norm_mat_tensor, labels_tensor = df_to_tensor(daily_forecast,norm_mat,labels)
    img_count = norm_mat_tensor.shape[0]
    split_type = 2 # Update for different split type(see data_split() function)
    x_train, y_train, full_trainset, full_testset = data_split(norm_mat_tensor, labels_tensor, split_type, split, k, img_count)
    training_sets_cv, validation_sets_cv = cross_val(x_train,y_train, k)

    print('Finished Dataprocessing')

    return training_sets_cv, validation_sets_cv, full_trainset, full_testset
