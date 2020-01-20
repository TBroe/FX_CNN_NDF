# script: preprocessing.py
# author: Tobias Broeckl
# date: 07.05.2019

# Script to pre-process the raw exchange rate data retrieved from "http://www.forextester.com/data/datasources".
# Includes shifting raw data by two hours and dealing with empty values.
# Raw data needs to transformed from .txt to .csv file and placed in ./data directory.
# Finally exports .csv files to be used as inputs for the model_selection and model_evaluation scripts.

import os
from datetime import timedelta, date
import argparse
import pandas as pd

os.chdir("../data")

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('-currency', choices=['EURUSD', 'USDJPY', 'GBPUSD'], type=str, default="EURUSD")

    opt = parser.parse_args()
    return opt

def main():
    opt = parse_arg()
    filename = opt.currency + ".csv"

    data = pd.read_csv(filename, sep=",", dtype={'<DTYYYYMMDD>': str, '<TIME>': str})
    data["datetime"] = data['<DTYYYYMMDD>'] + data['<TIME>']
    data['datetime'] = pd.to_datetime(data['datetime'], format="%Y%m%d%H%M%S")
    data.loc[:, 'day'] = data.loc[:, 'datetime'].dt.strftime('%a')
    data.rename(columns={'<CLOSE>': 'close'}, inplace=True)
    data = data.loc[:, ['datetime', 'day', 'close']]
    data_day = data.loc[data['datetime'].dt.minute == 0, :]
    data_day.loc[:, 'hour'] = data_day.loc[:, 'datetime'].dt.hour
    data_day.loc[:, 'datetime'] = data_day.loc[:, 'datetime'] + timedelta(hours=2)
    data_day.loc[:, 'hour'] = data_day.loc[:, 'datetime'].dt.hour
    data_day.loc[:, 'day'] = data_day.loc[:, 'datetime'].dt.strftime('%a')
    data_day.loc[:, 'delta_days'] = data_day.loc[:, 'datetime'].dt.date - date(2001, 1, 4)
    data_day.loc[:, 'delta_days'] = data_day.loc[:, 'delta_days'].dt.days
    name_day = opt.currency + "_clean.csv"
    data_day.to_csv(name_day)

if __name__ == '__main__':
    main()
