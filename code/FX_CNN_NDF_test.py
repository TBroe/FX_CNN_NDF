# Script: FX_CNN_NDF_train.py
# Author: Tobias Broeckl
# Date: 07.02.2019

# Script to evaluate the trained model on the testset.
# Testset is generated using the FX_CNN_NDF_data script.
# Includes testset generation, loading of model parameters and evaluation using AUC, accuracy and test loss.
# Note: The same parser arguments as for training should be used.

import argparse
import os

import FX_CNN_NDF_data
import FX_CNN_NDF_train as train
import torch
from torch.autograd import Variable
from sklearn.metrics import roc_curve
import csv


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument ('-class_method', type=int, default=2)
    parser.add_argument ('-currency', choices=['EURUSD', 'GBPJPY'], type=str, default='EURUSD')
    parser.add_argument ('-k', type=int, default=5)
    parser.add_argument ('-daily_forecast', action='store_true')
    parser.add_argument ('-batch_size', type=int, default=30)
    parser.add_argument ('-CNN_only', action='store_true')
    parser.add_argument ('-dropout_rate', type=float, default=0.3)
    parser.add_argument ('-n_class', type=int, default=2)
    parser.add_argument ('-epochs', type=int, default=1)
    parser.add_argument ('-periods_ahead', type=int, default=1)
    parser.add_argument ('-split', type=float, default=0.75)
    parser.add_argument ('-n_tree', type=int, default=30)
    parser.add_argument ('-tree_depth', type=int, default=5)
    parser.add_argument ('-tree_feature_rate', type=float, default=0.5)

    opt = parser.parse_args()
    return opt


def main():
    # Same arguments as training should be used.
    opt = parse_arg()

    # Generation of testset by calling prep_data() from FX_CNN_NDF_data script.
    # For dataset split type(chronological, random, testing_only) manually update prep_data() function before running.
    # Currency choice between EUR/USD and GBP/JPY.
    # Other inputs should be identical to trained model
    _, _, _, full_testset = FX_CNN_NDF_data.prep_data(opt.daily_forecast,
                                                       opt.periods_ahead,
                                                       opt.n_class,
                                                       opt.split,
                                                       opt.class_method,
                                                       opt.k,
                                                       opt.currency)


    os.chdir("../results")
    train.change_dir(opt.daily_forecast, opt.CNN_only, opt.n_class)

    # set up model and load parameters from local directory
    model = train.prepare_model(opt)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    trained_params = torch.load("trained_model", map_location=device)
    model.load_state_dict(trained_params)

    # TESTING
    model.eval()

    test_loader = train.dataloader(full_testset, opt.batch_size)

    probs = []
    targets = []
    preds = []

    running_loss = 0.0

    for batch_idx,(data, target) in enumerate(test_loader, 0):
        data = data.to(device)
        data = Variable(data)
        target = train.label_setup(target, device, opt.CNN_only)
        output = train.forward_pass(data, _ , model, training=False)
        _probs = train.get_probs(output, opt.CNN_only)
        _preds = _probs.data.max(1, keepdim=True)[1]
        preds.append(_preds)

        if opt.n_class == 2:
            class_one_prob = _probs[:, 1]
            probs.append(class_one_prob)

        _target = train.label_setup(target, device, opt.CNN_only)
        targets.append(_target)

        loss = train.model_loss(opt.CNN_only, output, target)
        running_loss += loss.item()

    # epoch end
    test_loss = running_loss /(batch_idx + 1)

    predictions = train.to_LongTensor(preds)
    labels = train.to_LongTensor(targets)

    if opt.n_class == 2:
        scores = torch.cat(probs)
        auc = train.auc_roc(scores, labels)
        conf_mat = train.get_conf_mat(predictions, labels)
        fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
        curve = [fpr, tpr]

    else:
        auc = "n/a"
        conf_mat = "n/a"
        curve = []

    # Evaluation metrics
    accuracy = train.get_accuracy(predictions, labels)

    with open("roc_curve.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerows(curve)

    if opt.n_class == 2:
        print("AUC under ROC Curve of testset: ", round(auc, 3))

    # Confusion Matrix

        print("Confusion Matrix: ", conf_mat)

    # Results
    print("Accuracy in testset: ", round(accuracy, 3))

    results_test = dict(auc=auc, accuracy=accuracy, test_loss=test_loss, conf_mat=conf_mat)

    with open('test_results.csv', 'w') as f:
        w = csv.DictWriter(f, results_test.keys())
        w.writeheader()
        w.writerow(results_test)

if __name__ == '__main__':
    main()
