# script: functions.py
# author: Tobias Broeckl
# date: 07.05.2019

# This script defines all functions used in model selection, model evaluation, figures, comb_gridsearch

import classes as models
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
import pandas as pd
import numpy as np
import itertools
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


def change_dir(CNN_only, currency, days_ahead):
    os.makedirs("../results", exist_ok=True)
    os.chdir("../results")

    if currency == "EURUSD":
        os.makedirs("./EURUSD", exist_ok=True)
        os.chdir("./EURUSD")
    elif currency == "USDJPY":
        os.makedirs("./USDJPY", exist_ok=True)
        os.chdir("./USDJPY")
    elif currency == "GBPUSD":
        os.makedirs("./GBPUSD", exist_ok=True)
        os.chdir("./GBPUSD")

    if CNN_only:
        os.makedirs("./CNN_only", exist_ok=True)
        os.chdir("./CNN_only")
    else:
        os.makedirs("./CNN_NDF", exist_ok=True)
        os.chdir("./CNN_NDF")

    if days_ahead==1:
        os.makedirs("./1d_ahead", exist_ok=True)
        os.chdir("./1d_ahead")

    elif days_ahead==5:
        os.makedirs("./5d_ahead", exist_ok=True)
        os.chdir("./5d_ahead")

    elif days_ahead==10:
        os.makedirs("./10d_ahead", exist_ok=True)
        os.chdir("./10d_ahead")


def get_loss(prediction, target, CNN_only):
    if CNN_only:
        loss = nn.CrossEntropyLoss()
        loss = loss(prediction, target)

    else:
        loss = F.nll_loss(torch.log(prediction), target)

    return loss


def prep_grid(CNN_only, grid_nr):
    epochs = [100,200,300]
    batch_size = [50, 200]
    n_tree = [50,100]
    tree_depth = [5,10]
    tree_feature_rate = [0.3,0.6]
    optim = ['sgd', 'adagrad']

    if CNN_only:
        grid = list(itertools.product(epochs, batch_size, optim))
        headers = ['epochs', 'batch_size', 'optimizer']
    else:
        grid = list(itertools.product(epochs, batch_size, n_tree, tree_depth, tree_feature_rate, optim))
        headers = ['epochs', 'batch_size', 'n_tree', 'tree_depth', 'feature_rate', 'optimizer']

    full_grid = pd.DataFrame(grid, columns=headers)

    if CNN_only:
        n_batches = int(len(full_grid) / 10)
    else:
        n_batches = int(len(full_grid) / 12)

    np.random.seed(0)
    rand_batch_idx = np.random.choice(full_grid.index.values, len(full_grid), replace=False)
    batches_idx = np.array_split(rand_batch_idx, n_batches)
    batch_idx = batches_idx[grid_nr]
    grid = full_grid.loc[batch_idx, :]
    grid.reset_index(inplace=True, drop=True)

    return grid


def prep_model(CNN_only, n_tree, tree_depth, tree_feature_rate):

    torch.manual_seed(0)

    if CNN_only:
        model= models.CNN_only()

    else:

        feat_layer = models.CNN_forest()

        forest = models.Forest(n_tree=n_tree,
                               tree_depth=tree_depth,
                               n_in_feature=64 * 3 * 3,
                               tree_feature_rate=tree_feature_rate)

        model = models.NeuralDecisionForest(feat_layer, forest)


    return model


def prep_optim(model, optimizer):
    params = [p for p in model.parameters() if p.requires_grad]

    if optimizer == "adagrad":
        optim = torch.optim.Adagrad(params, lr=1e-2)

    elif optimizer == "sgd":
        optim = torch.optim.SGD(params, lr=0.1)

    return optim


def get_conf_mat(predictions, class_labels):
    correct = predictions.eq(class_labels)

    tp = (predictions * correct).sum().item()
    fp = predictions.sum().item() - tp
    tn = correct.sum().item() - tp
    fn = predictions.shape[0] - tp - fp - tn

    conf_mat = dict(TP=tp, FP=fp, TN=tn, FN=fn)
    return conf_mat


def update_pi(model, feat_batches, target_batches, device):
    trees = model.forest.trees

    for tree in trees:
        mu_batches = []
        for feats in feat_batches:
            mu = tree(feats)
            mu_batches.append(mu)
        for _ in range(20):
            new_pi = torch.zeros((tree.n_leaf, 2))
            new_pi = new_pi.to(device)

            for mu, target in zip(mu_batches, target_batches):
                pi = tree.get_pi()
                prob = tree.cal_prob(mu, pi)
                pi = pi.data
                prob = prob.data
                mu = mu.data
                _target = target.unsqueeze(1)
                _pi = pi.unsqueeze(0)
                _mu = mu.unsqueeze(2)
                _prob = torch.clamp(prob.unsqueeze(1), min=1e-6, max=1.)

                _new_pi = torch.mul(torch.mul(_target, _pi), _mu) / _prob
                new_pi += torch.sum(_new_pi, dim=0)

            new_pi = F.softmax(new_pi, dim=1).data
            tree.update_pi(new_pi)


def target_setup(target, CNN_only):

    if CNN_only:
        target = target.view(-1)

    else:
        target = target.squeeze()

    if torch.cuda.is_available():
        target = target.type(torch.cuda.LongTensor)

    else:
        target = target.type(torch.LongTensor)

    return target


def to_bytetensor(list):

    tensor = torch.cat(list).squeeze()

    if torch.cuda.is_available():
        tensor = tensor.type(torch.cuda.ByteTensor)

    else:
        tensor = tensor.type(torch.ByteTensor)

    return tensor


def train_params(model, CNN_only, batch_size, trainset_val, device, optim):

    model.train()

    train_loader = torch.utils.data.DataLoader(trainset_val, batch_size=batch_size, shuffle=False, drop_last=True)

    if not CNN_only:

        # pi update
        feat_batches = []
        target_batches = []

        cls_onehot = torch.eye(2)

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target, cls_onehot = data.to(device), target.to(device), cls_onehot.to(device)
                target = target_setup(target, CNN_only)
                feats = model.feature_layer(data)
                feats = feats.view(feats.size()[0], -1)
                feat_batches.append(feats)
                target_batches.append(cls_onehot[target])

        update_pi(model, feat_batches, target_batches, device)

    # theta update
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader, 0):
        data, target = data.to(device), target.to(device)
        target = target_setup(target, CNN_only)
        optim.zero_grad()
        scores = model(data)
        loss = get_loss(scores, target, CNN_only)
        running_loss += loss.item()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],max_norm=5)
        optim.step()

    train_loss = running_loss / (batch_idx + 1)

    return train_loss


def validation(model, CNN_only, batch_size, valset, device):

    model.eval()

    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, drop_last=False)

    running_loss = 0.0

    probs_c1 = []
    predictions = []
    labels = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader, 0):
            data, target = data.to(device), target.to(device)
            targets = target_setup(target, CNN_only)
            scores = model(data)
            loss = get_loss(scores, targets, CNN_only)
            running_loss += loss.item()

            if CNN_only:
                probs = F.softmax(scores, dim=-1)
            else:
                probs = scores
            prob_c1_temp = probs[:,1]
            preds = probs.data.max(1, keepdim=True)[1]
            probs_c1.append(prob_c1_temp)
            predictions.append(preds)
            labels.append(targets)

    probs_c1=torch.cat(probs_c1).squeeze()
    predictions = to_bytetensor(predictions)
    labels = to_bytetensor(labels)

    conf_mat = get_conf_mat(predictions, labels)
    class_err = (conf_mat.get('FP') + conf_mat.get('FN')) / labels.shape[0]

    if torch.cuda.is_available():
        labels = labels.cpu().numpy()
        probs_c1 = probs_c1.cpu().numpy()

    auc = roc_auc_score(labels, probs_c1,)
    roc = roc_curve(labels, probs_c1, pos_label=1)
    val_loss = running_loss / (batch_idx + 1)

    return val_loss, class_err, roc, auc, conf_mat


def training(model, CNN_only, batch_size, epochs, device, optim, trainset, testset):
    for epoch in range(epochs):
        if (epoch + 1) % 10 == 0:
            print('- epoch:', epoch + 1)
        train_loss = train_params(model, CNN_only, batch_size, trainset, device, optim)
        test_loss, class_err, roc, auc, conf_mat = validation(model, CNN_only, batch_size, testset, device)

    return train_loss, test_loss, class_err, roc, auc, conf_mat


def prep_matrix(currency):
    filename = currency + "_clean.csv"
    data = pd.read_csv(filename, sep=",")
    data_mat = data.loc[:, ["hour", "delta_days", "close"]]
    data_mat.set_index(['hour', 'delta_days'], inplace=True, drop=True)
    data_mat = data_mat.unstack(level=1)
    data_mat.fillna(method='bfill', inplace=True)
    data_mat.fillna(method='ffill', inplace=True)
    data_mat = data_mat.stack()
    data_mat = data_mat.unstack(level=0)
    data_mat = pd.DataFrame(data_mat.values)

    return data_mat


def gen_delta(data_mat, days_ahead):

    medians = data_mat.median(axis=1)
    p = []

    for i in range(len(medians) - days_ahead):
        temp = (data_mat.loc[(i + 1):(i + days_ahead), :] / medians[i]) - 1
        p.append(np.median(temp.values))

    p = pd.Series(p)

    return p


def classify(p):
    bins = [-1, 0, 1]
    labels = [0, 1]
    classes = pd.cut(p, bins=bins, include_lowest=True, labels=labels)

    return classes


def batching(data_mat, y_classes, days_ahead):
    y_batches = []
    x_batches = []

    x_batch_length = 24

    for i in range(data_mat.shape[0] - x_batch_length + 1 - days_ahead):
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
        x_batch_norm = (x_batches[i] - min) / (max - min)
        x_batches_norm.append(x_batch_norm)

    return x_batches_norm


def df_to_tensor(x_batches_norm, y_batches):
    x_batches = torch.from_numpy(np.asarray(x_batches_norm)).type(torch.FloatTensor).view(-1, 24, 24)
    y_batches = torch.from_numpy(np.asarray(y_batches)).type(torch.IntTensor).view(-1, 1, 1)

    return x_batches, y_batches


def test_data(x_batches, y_batches, split, img_count):
    end_trainset = int(img_count * split)
    end_testset = img_count

    index = np.arange(end_testset)
    train_idx = np.argwhere(index <= end_trainset)
    test_idx = np.argwhere(index > end_trainset)

    x_trainset = x_batches[train_idx, :, :]
    y_trainset = y_batches[train_idx, :, :]

    x_testset = x_batches[test_idx, :, :]
    y_testset = y_batches[test_idx, :, :]

    trainset_full = TensorDataset(x_trainset, y_trainset)
    testset = TensorDataset(x_testset, y_testset)

    return trainset_full, testset


def validation_data(x_batches, y_batches, split, img_count):
    end_trainset_val = int(img_count * split * split)
    end_valset = int(img_count * split)

    index = np.arange(end_valset)
    train_idx_val = np.argwhere(index <= end_trainset_val)
    val_idx = np.argwhere(index > end_trainset_val)

    x_trainset_val = x_batches[train_idx_val, :, :]
    y_trainset_val = y_batches[train_idx_val, :, :]

    x_valset = x_batches[val_idx, :, :]
    y_valset = y_batches[val_idx, :, :]

    trainset_val = TensorDataset(x_trainset_val, y_trainset_val)
    valset = TensorDataset(x_valset, y_valset)

    return trainset_val, valset


def prep_data(currency, days_ahead):
    os.chdir("../data")
    data_mat = prep_matrix(currency)
    delta = gen_delta(data_mat, days_ahead)
    labels = classify(delta)
    non_norm_mat, labels = batching(data_mat, labels, days_ahead)
    norm_mat = min_max_norm(non_norm_mat)
    norm_mat_tensor, labels_tensor = df_to_tensor(norm_mat, labels)
    img_count = norm_mat_tensor.shape[0]

    return norm_mat_tensor, labels_tensor, img_count
