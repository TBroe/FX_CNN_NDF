# script: figures.py
# author: Tobias Broeckl
# date: 07.05.2019

# Script that generates a single figure that compares the performance of the deep CNN and deep NDF across all
# currency pairs for a defined prediction horizon, which is defined in the -days_ahead argument.

import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import matplotlib as mpl
import pandas as pd
import numpy as np

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('-days_ahead', type=int)
    opt = parser.parse_args()

    return opt

os.chdir("../results")

def main():
    mpl.rc('text', usetex=True)
    mpl.rc('font', family='serif')

    opt = parse_arg()
    days_ahead = "./" + str(opt.days_ahead) + "d_ahead"

    currencies = ['EURUSD','USDJPY','GBPUSD']

    auc_cnn = []
    acc_cnn = []

    auc_ndf = []
    acc_ndf = []

    tpr_cnn = []
    fpr_cnn = []

    tpr_ndf = []
    fpr_ndf = []

    path = os.getcwd()

    for i in currencies:

        os.chdir(path)

        # CNN - results
        os.chdir("./" + str(i))
        os.chdir("./CNN_only")
        os.chdir(days_ahead)

        roc_cnn = pd.read_csv("roc_curve.csv", sep=",", header=None)
        test_results_cnn = pd.read_csv("test_results.csv", sep=",")

        auc_cnn.append(round(float(test_results_cnn.auc),2))
        acc_cnn.append(round(float(test_results_cnn.accuracy),2))

        fpr_cnn.append(roc_cnn.iloc[0, :])
        tpr_cnn.append(roc_cnn.iloc[1, :])

        # NDF - results
        os.chdir(path)
        os.chdir("./" + str(i))
        os.chdir("./CNN_NDF")
        os.chdir(days_ahead)

        roc_ndf = pd.read_csv("roc_curve.csv", sep=",", header=None)
        test_results_ndf = pd.read_csv("test_results.csv", sep=",")

        auc_ndf.append(round(float(test_results_ndf.auc),2))
        acc_ndf.append(round(float(test_results_ndf.accuracy),2))

        fpr_ndf.append(roc_ndf.iloc[0, :])
        tpr_ndf.append(roc_ndf.iloc[1, :])

    os.chdir(path)
    os.makedirs("./figures", exist_ok=True)
    os.chdir("./figures")

    os.makedirs(days_ahead, exist_ok=True)
    os.chdir(days_ahead)

    currency_names=['EUR/USD','USD/JPY','GBP/USD']

    # plots
    fig,axes=plt.subplots(2,3, figsize=(11,8), sharey='row')
    #fig.suptitle("prediction horizon " +str(opt.days_ahead)+ " day ahead", fontsize=16)
    plt.matplotlib.pyplot.subplots_adjust(wspace=0.3)
    plt.matplotlib.pyplot.subplots_adjust(hspace=0.1)
    for i in range(len(currencies)):

        axes[0, 0].set_ylabel("accuracy")

        axes[0,i].set_title(currency_names[i], fontsize='large')
        axes[0,i].tick_params(axis='both', labelsize='small')
        axes[0,i].set_ylim(0, 1)
        axes[0,i].axhline(0.5,0, 1, color='grey', lw=0.8, linestyle="--")
        axes[0,i].yaxis.set_major_formatter(PercentFormatter(1))
        axes[0,i].bar("deep CNN", acc_cnn[i], alpha=0.7)
        axes[0,i].text("deep CNN", acc_cnn[i] + 0.05, str(round((100* acc_cnn[i]))) + "\%", ha="center",
                     fontsize='small')
        axes[0,i].bar("deep NDF", acc_ndf[i],alpha=0.7)
        axes[0,i].text("deep NDF", acc_ndf[i] + 0.05, str(round((100 * acc_ndf[i]))) + "\%", ha="center",
                     fontsize='small')

        axes[1, 0].set_ylabel("True Positive Rate")
        axes[1,i].tick_params(axis='both',labelsize='small')
        axes[1,i].plot(fpr_cnn[i], tpr_cnn[i], label="ROC curve CNN", lw=0.8)
        axes[1,i].text(0,0.95,"AUC CNN: "+str('{:03.2f}'.format(round(auc_cnn[i],3))))
        axes[1,i].plot(fpr_ndf[i], tpr_ndf[i], label="ROC curve NDF", lw=0.8)
        axes[1,i].text(0,0.87, "AUC NDF: " + str('{:03.2f}'.format(round(auc_ndf[i], 3))))
        axes[1,i].plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), color="grey", lw=0.8,linestyle="--")
        axes[1,i].set_xlabel("False Positive Rate")
        axes[1,i].legend(loc="lower right", fontsize='small')

    fig.tight_layout()
    plt.show()
    name="cnn_ndf_comp.png"
    fig.savefig(name, bbox_inches='tight')

if __name__ == '__main__':
    main()
