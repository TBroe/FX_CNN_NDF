# Script: FX_CNN_NDF_data_preprocessing
# Author: Tobias Broeckl
# Date: 07.02.2019

# Script to pre-process the raw exchange rate data retrieved from "http://www.forextester.com/data/datasources".
# Includes shifting data in raw data by two hours (reasoning in Jupyter Notebook file) and dealing with NaNs.
# Raw data needs to transformed from .txt to .csv file and placed in local directory.
# Finally exports .csv files to be used as input for the FX_CNN_NDF_data script.
# Both day/hour and hour/minute are supported.
# Use parser to select between EUR/USD and GBP/JPY.

import os
from datetime import timedelta, date
import argparse
import pandas as pd

os.chdir("../data") # Use local directory to import raw data as .csv files and later export output files.

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('-currency', choices=['EURUSD','GBPJPY'], type=str, default='EURUSD')

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

    # Day/Hour Setting
    data_day = data.loc[data['datetime'].dt.minute == 0, :]
    data_day.loc[:, 'hour'] = data_day.loc[:, 'datetime'].dt.hour
    data_day.loc[:, "datetime"] = data_day.loc[:, "datetime"] + timedelta(hours=2)
    data_day.loc[:, "hour"] = data_day.loc[:, "datetime"].dt.hour
    data_day.loc[:, 'day'] = data_day.loc[:, 'datetime'].dt.strftime('%a')
    data_day.loc[:,'delta_days']=data_day.loc[:,'datetime'].dt.date-date(2001,1,4)
    data_day.loc[:,"delta_days"]=data_day.loc[:,"delta_days"].dt.days
    name_day = opt.currency + "_prep_day.csv"
    data_day.to_csv(name_day)

    # Hour/Minute Setting
    data_hour = data
    data_hour.loc[:, "datetime"] = data_hour.loc[:, "datetime"] + timedelta(hours=2)
    data_hour.loc[:, 'day'] = data_day.loc[:, 'datetime'].dt.strftime('%a')
    data_hour.loc[:, 'hour'] = data_hour.loc[:, 'datetime'].dt.hour
    data_hour.loc[:, 'minute'] = data_hour.loc[:, 'datetime'].dt.minute
    data_hour.loc[:, 'date'] = data_hour.loc[:, 'datetime'].dt.date
    name_hour = opt.currency + "_prep_hour.csv"
    data_hour.to_csv(name_hour)

if __name__ == '__main__':
    main()