# script: model_evaluation.py
# author: Tobias Broeckl
# date: 07.05.2019

# Script used to train a single model on the full training data with a specified set of hyperparameter values.
# The trained model is then evaluated on the test set.
# The script returns .csv files of the model evaluation results in a generated ./results directory.

import argparse
import torch
import functions as func
import csv
import sys

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('-CNN_only', action='store_true',default=False)
    parser.add_argument('-days_ahead', type=int, default=1)
    parser.add_argument('-currency', choices=['EURUSD', 'USDJPY', 'GBPUSD'], type=str, default="EURUSD")
    parser.add_argument('-split', type=float, default=0.8)
    parser.add_argument('-optimizer',choices=['adagrad','sgd'],type=str, default="sgd")
    parser.add_argument('-batch_size',type=int,default=200)
    parser.add_argument('-epochs',type=int,default=10)
    parser.add_argument('-n_tree',type=int,default=50)
    parser.add_argument('-tree_depth',type=int,default=5)
    parser.add_argument('-tree_feature_rate',type=float,default=0.3)
    opt = parser.parse_args()

    return opt

def main():
    opt = parse_arg()
    print('')
    print(opt)
    x_batches, y_batches, img_count = func.prep_data(opt.currency, opt.days_ahead)
    trainset_full, testset = func.test_data(x_batches, y_batches, opt.split, img_count)

    func.change_dir(opt.CNN_only, opt.currency, opt.days_ahead)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = func.prep_model(opt.CNN_only,
                         n_tree=opt.n_tree,
                         tree_depth=opt.tree_depth,
                         tree_feature_rate=opt.tree_feature_rate)

    model = model.to(device)

    optim = func.prep_optim(model, opt.optimizer)

    train_loss, val_loss, class_err, roc, auc, conf_mat = func.training(model,
                                                                        opt.CNN_only,
                                                                        opt.batch_size,
                                                                        opt.epochs,
                                                                        device,
                                                                        optim,
                                                                        trainset_full,
                                                                        testset)
    accuracy = 1-class_err

    print("accuracy on testset", round(100* accuracy), "%")
    print("AUC testset", round(auc,3))
    print("confusion matrix", conf_mat)

    test_results = dict(auc=auc, accuracy=accuracy)

    with open('test_results.csv', 'w') as f:
        w = csv.DictWriter(f, test_results.keys())
        w.writeheader()
        w.writerow(test_results)

    with open('conf_mat.csv', 'w') as f:
        w = csv.DictWriter(f, conf_mat.keys())
        w.writeheader()
        w.writerow(conf_mat)

    tpr = roc[0]
    fpr = roc[1]
    roc_curve = [tpr,fpr]

    with open("roc_curve.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerows(roc_curve)

    # Print the console output
    sys.stout = open("arguments.txt", "w")
    print(opt, file=sys.stout)

if __name__ == '__main__':
    print(__file__)
    main()
