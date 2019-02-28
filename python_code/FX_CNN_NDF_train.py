# Script: FX_CNN_NDF_train
# Author: Tobias Broeckl
# Date: 07.02.2019

# Script used to perform k-fold cross-validation and training of parameters for -
# standalone CNN and combination of CNN and Neural Decision Forest.
# This script is calling functions from the scripts: FX_CNN_NDF_data.py and FX_CNN_NDF_model.py

import argparse
import csv
import os
from datetime import datetime
import sys

import FX_CNN_NDF_data as data
import FX_CNN_NDF_model
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.autograd import Variable

startTime = datetime.now()


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('-class_method',type=int,default=2)
    parser.add_argument('-currency',choices=['EURUSD','GBPJPY'],type=str,default='EURUSD')
    parser.add_argument('-k',type=int,default=5)
    parser.add_argument('-daily_forecast',action='store_true')
    parser.add_argument('-batch_size',type=int,default=30)
    parser.add_argument('-CNN_only',action='store_true')
    parser.add_argument('-dropout_rate',type=float,default=0.3)
    parser.add_argument('-n_class',type=int,default=2)
    parser.add_argument('-epochs',type=int,default=1)
    parser.add_argument('-periods_ahead',type=int,default=1)
    parser.add_argument('-split',type=float,default = 0.75)
    parser.add_argument('-n_tree',type=int,default=30)
    parser.add_argument('-tree_depth',type=int,default=5)
    parser.add_argument('-tree_feature_rate',type=float,default=0.5)

    opt = parser.parse_args()
    return opt


def change_dir(daily_forecast,CNN_only,n_class):
    """function to change directories used for training output. If necessary,directories are generated"""
    os.makedirs("../results",exist_ok=True)
    os.chdir(
        "../results")  # as FX_CNN_NDF_data.prep_data() is called,the dir is currently data. Change to results for what is to follow.

    if n_class == 2:
        os.makedirs("./binary",exist_ok=True)
        os.chdir("./binary")
    else:
        os.makedirs("./multiclass",exist_ok=True)
        os.chdir("./multiclass")

    if daily_forecast:
        os.makedirs("./daily",exist_ok=True)
        os.chdir("./daily")
    else:
        os.makedirs("./hourly",exist_ok=True)
        os.chdir("./hourly")

    if CNN_only:
        os.makedirs("./CNN_only",exist_ok=True)
        os.chdir("./CNN_only")
    else:
        os.makedirs("./CNN_NDF",exist_ok=True)
        os.chdir("./CNN_NDF")


def export(dictionary,multiple):
    """function to export training results that are stored in dictionaries to .csv files."""
    for name in dictionary:
        temp = name + ".csv"
        with open(temp,'w') as f:
            writer = csv.writer(f)
            if multiple:
                writer.writerows(dictionary[name])
            else:
                writer.writerow(dictionary[name])


def prepare_model(opt):
    """function to generate the model structure based on parset arguments that are stored in the "opt" object.
    This function references classes from the FX_CNN_NDF_model script"""
    feat_layer = FX_CNN_NDF_model.FX_FeatureLayer(opt.dropout_rate,opt.CNN_only,opt.n_class,opt.daily_forecast)

    if not opt.CNN_only:

        forest = FX_CNN_NDF_model.Forest(n_tree=opt.n_tree,
                                          tree_depth=opt.tree_depth,
                                          n_in_feature=64 * 3 * 3,
                                          tree_feature_rate=opt.tree_feature_rate,
                                          n_class=opt.n_class)

        model = FX_CNN_NDF_model.NeuralDecisionForest(feat_layer,forest)

    else:
        model = feat_layer

    return model


def prepare_optim(model):
    """initiates the optimizer Adam with predefined learning rate and weight decay for the model"""
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(params,lr=0.001,weight_decay=1e-5)


def dataloader(data,batch_size):
    """function to return a dataloader object for training and valiadtion with given batch size."""
    dataloader = torch.utils.data.DataLoader(data,batch_size=batch_size,shuffle=False)
    return dataloader


def label_setup(target,device,CNN_only):
    """function to format the label values to be used in both the CNN_only and CNN_NDF setting."""
    target = target.to(device)
    target = Variable(target)

    if CNN_only:
        target = target.view(-1)  # added

    else:
        target = target.squeeze()

    if torch.cuda.is_available():
        target = target.type(torch.cuda.LongTensor)
    else:
        target = target.type(torch.LongTensor)

    return target


def update_pi(n_class,model,train_loader,device):
    """function to update pi for the CNN_NDF training option"""
    cls_onehot = torch.eye(n_class)
    feat_batches = []
    target_batches = []

    for batch_idx,(data,target) in enumerate(train_loader):

        data = data.to(device)
        data = Variable(data)

        target = label_setup(target,device,CNN_only = False)

        if torch.cuda.is_available():
            feats = model.module.feature_layer(data)
        else:
            feats = model.feature_layer(data)

        feats = feats.view(feats.size()[0],-1)
        feat_batches.append(feats)

        cls_onehot = cls_onehot.to(device)
        target_batches.append(cls_onehot[target])

    ## Update the \pi for each tree
    if torch.cuda.is_available():
        trees = model.module.forest.trees
    else:
        trees = model.forest.trees

    for tree in trees:
        mu_batches = []
        for feats in feat_batches:
            mu = tree(feats)  # [batch_size,n_leaf]
            mu_batches.append(mu)
        for _ in range(20):
            new_pi = torch.zeros((tree.n_leaf,tree.n_class))  # Tensor [n_leaf,n_class]

            new_pi = new_pi.to(device)

            for mu,target in zip(mu_batches,target_batches):
                pi = tree.get_pi()  # [n_leaf,n_class]
                prob = tree.cal_prob(mu,pi)  # [batch_size,n_class]

                # NDF - Variable to Tensor
                pi = pi.data
                prob = prob.data
                mu = mu.data

                _target = target.unsqueeze(1)  # [batch_size,1,n_class]
                _pi = pi.unsqueeze(0)  # [1,n_leaf,n_class]
                _mu = mu.unsqueeze(2)  # [batch_size,n_leaf,1]
                _prob = torch.clamp(prob.unsqueeze(1),min=1e-6,max=1.)  # [batch_size,1,n_class]

                _new_pi = torch.mul(torch.mul(_target,_pi),_mu) / _prob  # [batch_size,n_leaf,n_class]
                new_pi += torch.sum(_new_pi,dim=0)

            new_pi = F.softmax(Variable(new_pi),dim=1).data
            tree.update_pi(new_pi)


def forward_pass(data,optim,model,training):
    """function to forward pass data through the model"""
    if training:
        optim.zero_grad()
        output = model(data)

    else:
        with torch.no_grad():
            output = model(data)

    return output


def backward_pass(loss,model,optim):
    """function to backward pass updates through the model using the optimzer,model and loss value as inputs."""
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],max_norm=5)
    optim.step()


def model_loss(CNN_only,output,target):
    """function to calculate the model loss value in both the CNN_only and CNN_NDF setting."""
    if CNN_only:
        loss = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()
        loss = loss(output,target)

    else:
        loss = F.nll_loss(torch.log(output),target)

    return loss


def get_probs(output,CNN_only):
    """function to return class probabilities from model output using a Softmax function in the CNN_only case."""
    if CNN_only:
        probs = F.softmax(output,dim=1)

    else:
        probs = output
    return probs


def get_accuracy(predictions,class_labels):
    """ function to return the accuracy as a float given the prediction value and corresponding class label as inputs"""

    correct_count = predictions.eq(class_labels).sum().item()
    total_count = len(class_labels)
    accuracy = correct_count / total_count

    return accuracy


def get_conf_mat(predictions,class_labels):
    """function to return the confusion matrix with predictions and class labels as inputs. 
    Used only in n_class == 2 setting."""

    correct = predictions.eq(class_labels)
    if torch.cuda.is_available():
        correct = correct.type(torch.cuda.LongTensor)
    else:
        correct = correct.type(torch.LongTensor)

    tp =(predictions * correct).sum().item()
    fp = predictions.sum().item() - tp
    tn = correct.sum().item() - tp
    fn = predictions.shape[0] - tp - fp - tn

    conf_mat = [tp,fp,tn,fn]
    return conf_mat


def auc_roc(scores,class_labels):
    """function to return the auc score given the predictions scores of class 1 and the respective class labels. 
    Used only in n_class == 2 setting."""

    if torch.cuda.is_available():
        class_labels = class_labels.cpu().numpy()
        scores = scores.cpu().numpy()

    auc = roc_auc_score(class_labels,scores)

    return auc


def to_LongTensor(list):
    """function to generate a long tensor from a list."""
    tensor = torch.cat(list).squeeze()

    if torch.cuda.is_available():
        tensor = tensor.type(torch.cuda.LongTensor)

    else:
        tensor = tensor.type(torch.LongTensor)

    return tensor


def train(model,training_sets_cv,validation_sets_cv,full_trainset,opt,device):

    # Objects to collect info for across CV fold
    training_loss_cv = []
    validation_loss_cv = []
    accuracy_cv = []
    auc_cv = []
    conf_matrices_cv = []

    k = opt.k  # number of folds(min = 2)

    for i in range(k + 1):  # Cross-validation plus final loop whole training set

        # loading the initial model parameters before each CV-fold and full training

        model.load_state_dict(torch.load("init"))
        optim = prepare_optim(model)

        if i <= k - 1:
            print("CV fold:",i + 1)
        else:
            print("Training on full training set")

        if torch.cuda.is_available():
            import pycuda.driver as cuda
            cuda.init()

        # Objects that are reset after each fold. Collect data across epochs.

        training_loss = []
        validation_loss = []
        training_loss_full = []

        # Training set-up

        for epoch in range(opt.epochs):

            model.train ()

            # Objects that are reset after each epoch. Collect data for single epoch.
            probs = []
            targets = []
            preds = []

            if i <= k - 1:
                data = training_sets_cv[i]

            else:
                data = full_trainset

            train_loader = dataloader (data,opt.batch_size)

            if not opt.CNN_only:

                update_pi(opt.n_class,model,train_loader,device)

            running_loss = 0.0

            for batch_idx,(data,target) in enumerate(train_loader,0):
                data = data.to(device)
                data = Variable(data)

                target = label_setup(target,device,opt.CNN_only)
                output = forward_pass(data,optim,model,training=True)
                loss = model_loss(opt.CNN_only,output,target)
                running_loss += loss.item()
                backward_pass(loss,model,optim)

            train_loss = running_loss /(batch_idx + 1)

            if i <= k - 1:
                training_loss.append(train_loss)
            else:
                training_loss_full.append(train_loss)

            # VALIDATION
            model.eval()

            if i <= k - 1:

                test_loader = dataloader(validation_sets_cv[i],opt.batch_size)

                running_loss = 0.0

                for batch_idx,(data,target) in enumerate(test_loader,0):
                    data = data.to(device)
                    data = Variable(data)
                    target = label_setup(target,device,opt.CNN_only)
                    output = forward_pass(data,optim,model,training=False)
                    _probs = get_probs(output,opt.CNN_only)
                    _preds = _probs.data.max(1,keepdim=True)[1]
                    preds.append(_preds)

                    if opt.n_class == 2:
                        class_one_prob = _probs[:,1]
                        probs.append(class_one_prob)

                    _target = label_setup(target,device,opt.CNN_only)
                    targets.append(_target)

                    loss = model_loss(opt.CNN_only,output,target)
                    running_loss += loss.item()

                # epoch end
                val_loss = running_loss/(batch_idx + 1)
                validation_loss.append(val_loss)

                predictions = to_LongTensor(preds)
                labels = to_LongTensor(targets)

                if opt.n_class == 2:
                    scores = torch.cat(probs)

                    auc = auc_roc(scores,labels)
                    conf_mat = get_conf_mat(predictions,labels)

                else:
                    auc = "n/a"
                    conf_mat = "n/a"

                #Evaluation metrics
                accuracy = get_accuracy(predictions,labels)

        # CV fold end
        if i <= k - 1:
            training_loss_cv.append(training_loss)
            validation_loss_cv.append(validation_loss)
            accuracy_cv.append(accuracy)
            conf_matrices_cv.append(conf_mat)

            if opt.n_class == 2:
                auc_cv.append(auc)

    return training_loss_cv,validation_loss_cv,accuracy_cv,auc_cv,conf_matrices_cv,training_loss_full


def main():
    opt = parse_arg()
    print(opt)

    training_sets_cv,validation_sets_cv,full_trainset,_ = data.prep_data \
           (
            opt.daily_forecast,
            opt.periods_ahead,
            opt.n_class,
            opt.split,
            opt.class_method,
            opt.k,
            opt.currency
        )

    change_dir(opt.daily_forecast,opt.CNN_only,opt.n_class)

    model = prepare_model(opt)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        model = nn.DataParallel(model)

    model.to(device)
    torch.save(model.state_dict(),"init")

    # Saving the model's initial parameters to be reused for cross-validation

    training_loss_cv,\
    validation_loss_cv,\
    accuracy_cv,\
    auc_cv,\
    conf_matrices_cv,\
    training_loss_full = train \
           (
            model,
            training_sets_cv,
            validation_sets_cv,
            full_trainset,
            opt,
            device
        )

    print("runtime complete:",datetime.now() - startTime)

    # EXPORT model parameters and training results

    # Model parameters
    if torch.cuda.is_available():
        torch.save(model.module.state_dict(),"trained_model")
    else:
        torch.save(model.state_dict(),"trained_model")

    # Results

    # Lists
    auc_cv = list(np.array(auc_cv))
    training_loss_full = list(np.array(training_loss_full))
    accuracy_cv = list(np.array(accuracy_cv))

    results_list = dict(auc_cv = auc_cv,
                        accuracy_cv = accuracy_cv,
                        training_loss_full = training_loss_full)

    export(results_list,multiple = False)
    # Lists of lists

    results_lol = dict(training_loss_cv = training_loss_cv,
                       validation_loss_cv = validation_loss_cv,
                       confusion_matrices_cv = conf_matrices_cv)
    export(results_lol,multiple = True)

    # Print the console output
    sys.stout = open("arguments.txt","w")
    print(opt,file=sys.stout)


if __name__ == '__main__':
    print(__file__)
    main()
