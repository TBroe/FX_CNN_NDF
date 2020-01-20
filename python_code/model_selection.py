# script: model_selection.py
# author: Tobias Broeckl
# date: 07.05.2019

# Script that employs grid search for model selection. The tested hyperparameter values are defined in the
# functions script in the function prep_grid(). The script returns a .csv file with the results of the trialed
# hyperparameter values in a generated ./results directory.

import argparse

import functions as func 
import torch
import pandas as pd
from datetime import datetime

startTime = datetime.now()


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('-CNN_only', action='store_true')
    parser.add_argument('-days_ahead', type=int)
    parser.add_argument('-currency', choices=['EURUSD', 'USDJPY', 'GBPUSD'], type=str, default="EURUSD")
    parser.add_argument('-split', type=float, default=0.8)
    parser.add_argument('-grid_nr', type=int)
    opt = parser.parse_args()

    return opt


def main():
    opt = parse_arg()
    print('')
    print('CNN_only:',opt.CNN_only, ', currency:', opt.currency, ', train/valset split:', opt.split)

    x_batches, y_batches, img_count = func.prep_data(opt.currency, opt.days_ahead)
    trainset_val, valset = func.validation_data(x_batches, y_batches, opt.split, img_count)

    func.change_dir(opt.CNN_only, opt.currency, opt.days_ahead)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    grid = func.prep_grid(opt.CNN_only, opt.grid_nr)
    print('')
    print("testing", grid.shape[0], "hyperparameter combinations in gridsearch")
    results = []

    for i in range(len(grid)):
        hyparams = grid.loc[i]
        print('')
        print(grid.loc[[i]])
        batch_size = int(hyparams['batch_size'])
        epochs = int(hyparams['epochs'])
        optimizer = str(hyparams['optimizer'])

        if opt.CNN_only:
            n_tree = 'n/a'
            tree_depth = 'n/a'
            tree_feature_rate = 'n/a'
        else:
            n_tree = int(hyparams['n_tree'])
            tree_depth = int(hyparams['tree_depth'])
            tree_feature_rate = float(hyparams['feature_rate'])

        model = func.prep_model(opt.CNN_only,
                             n_tree=n_tree,
                             tree_depth=tree_depth,
                             tree_feature_rate=tree_feature_rate)

        model = model.to(device)
        optim = func.prep_optim(model, optimizer)

        train_loss, val_loss, class_err, roc, auc, conf_mat = func.training(model,
                                                                            opt.CNN_only,
                                                                            batch_size,
                                                                            epochs,
                                                                            device,
                                                                            optim,
                                                                            trainset_val,
                                                                            valset)

        result_temp = [train_loss, val_loss, class_err, auc]
        print("class_err", round(class_err, 3), ", AUC:", round(auc, 3))
        results.append(result_temp)

    header_results = ['train_loss', 'val_loss', 'class_err', 'auc']
    results = pd.DataFrame(results, columns=header_results)

    grid_results = pd.concat([grid, results], axis=1)
    grid_results = grid_results.sort_values("class_err", ascending=True)
    name = "val_results_batch_" + str(opt.grid_nr) + ".csv"
    grid_results.to_csv(name, encoding='utf-8', index=False)

    print("runtime complete:", datetime.now() - startTime)

if __name__ == '__main__':
    print(__file__)
    main()
