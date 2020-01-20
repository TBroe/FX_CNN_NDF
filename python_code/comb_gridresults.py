# script: comb_gridsearch.py
# author: Tobias Broeckl
# date: 07.05.2019

# Script that combines the individual grid search results that are generated from the different grid_nr in the
# model_selection script. The final grid search result is ranked by classification error and is saved in
# the ./results directory. Please note that the model_selection script for one model and prediction horizon needs to
# be run with all possible grid_nr arguments before comb_gridsearch.py can be used.

import pandas as pd
import argparse
import glob
import os
import functions as func

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('-currency', choices=['EURUSD', 'USDJPY', 'GBPUSD'], type=str, default="EURUSD")
    parser.add_argument('-CNN_only', action='store_true')
    parser.add_argument('-days_ahead', type=int, default=1)
    opt = parser.parse_args()

    return opt

opt = parse_arg()

func.change_dir(opt.CNN_only, opt.currency, opt.days_ahead)

path = os.getcwd()
all_results = glob.glob(path + '/*.csv')

try:
    all_results.remove(path + '/val_results_full.csv')
except:
    pass

try:
    all_results.remove(path + '/roc_curve.csv')
except:
    pass

try:
    all_results.remove(path + '/conf_mat.csv')
except:
    pass

try:
    all_results.remove(path + '/test_results.csv')
except:
    pass

li = []

for filename in all_results:
    df_temp = pd.read_csv(filename, index_col=None, header=0)
    li.append(df_temp)

full_results = pd.concat(li, axis=0, ignore_index=True)

result_sorted = full_results.sort_values('class_err', ascending=True)
result_sorted.reset_index(inplace=True, drop=True)
result_sorted.index += 1
result_sorted.index.name = 'rank'
pd.set_option("display.max_columns", 10)

print("top 10 hyperparameter combinations by classificiation error resulting from gridsearch")
print(result_sorted.head(10))
result_sorted.to_csv('val_results_full.csv')
