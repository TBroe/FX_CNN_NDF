# Script: FX_CNN_NDF_results
# Author: Tobias Broeckl
# Date: 07.02.2019

# Script for visualization of results from CNN_only and NDF.
# Please see corresponding jupyter notebook script in directory ./code/jupyter for more details.


import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np

os.chdir("../results")  # as FX_CNN_NDF_data.prep_data() is called, the directory is currently ./data.


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('-daily_forecast', action='store_true')
    parser.add_argument('-n_class', type=int, default=2)

    opt = parser.parse_args()
    return opt


def main():
    opt = parse_arg()

    if opt.n_class == 2:
        os.chdir("./binary")
    else:
        os.chdir("./multiclass")

    if opt.daily_forecast:
        os.chdir("./daily")
    else:
        os.chdir("./hourly")

    path = os.getcwd()

    os.chdir("./CNN_only")

    # Import results

    training_loss_cv_CNN = pd.read_csv("training_loss_cv.csv", sep=",", header=None)
    validation_loss_cv_CNN = pd.read_csv("validation_loss_cv.csv", sep=",", header=None)

    accuracy_cv_CNN = pd.read_csv("accuracy_cv.csv", sep=",", header=None).T
    training_loss_full_CNN = pd.read_csv("training_loss_full.csv", sep=",", header=None)
    test_results_CNN = pd.read_csv("test_results.csv", sep=",")
    if opt.n_class ==2:
        auc_cv_CNN = pd.read_csv ("auc_cv.csv", sep=",", header=None).T
        roc_CNN = pd.read_csv("roc_curve.csv", sep=",", header=None)

    os.chdir(path)
    os.chdir("./CNN_NDF")

    training_loss_cv_NDF = pd.read_csv("training_loss_cv.csv", sep=",", header=None)
    validation_loss_cv_NDF = pd.read_csv("validation_loss_cv.csv", sep=",", header=None)
    accuracy_cv_NDF = pd.read_csv("accuracy_cv.csv", sep=",", header=None).T
    training_loss_full_NDF = pd.read_csv("training_loss_full.csv", sep=",", header=None)
    test_results_NDF = pd.read_csv("test_results.csv", sep=",")

    if opt.n_class == 2:
        auc_cv_NDF = pd.read_csv("auc_cv.csv", sep=",", header=None).T
        roc_NDF = pd.read_csv("roc_curve.csv", sep=",", header=None)
    #
    os.chdir (path)
    os.makedirs("./figures", exist_ok=True)
    os.chdir("./figures")

    # Training / validation loss CNN_only vs. NDF

    if training_loss_cv_CNN.shape[0] == 10:
        fig, axes = plt.subplots(2, int(training_loss_cv_CNN.shape[0] / 2 + 0.5))
        axes.ravel()
    else:
        fig, axes = plt.subplots(1, int(training_loss_cv_CNN.shape[0]), figsize=(30, 7), sharey=True)
        fig.text(0.1, 0.5, 'loss value', va='center', rotation='vertical')

    fig.suptitle("Training / Validation loss (Cross-Validation)", fontsize=20)
    cnn_only = mpatches.Patch(color='C0', label='CNN')
    cnn_ndf = mpatches.Patch(color='C1', label='CNN_NDF')
    plt.legend(handles=[cnn_only, cnn_ndf])

    if training_loss_cv_CNN.shape[0] == training_loss_cv_NDF.shape[0]:
        folds = training_loss_cv_CNN.shape[0]

    else:
        print("Different amount of CV folds used in training")
        raise NotImplementedError

    epochs_cnn = range(training_loss_cv_CNN.shape[1])
    epochs_ndf = range(training_loss_cv_NDF.shape[1])

    for i in range(folds):
        axes[i].plot(epochs_cnn, training_loss_cv_CNN.loc[i, :], "-", color="C0")
        axes[i].plot(epochs_ndf, training_loss_cv_NDF.loc[i, :], "-", color="C1")
        axes[i].plot(epochs_cnn, validation_loss_cv_CNN.loc[i, :], "-", color="C0")
        axes[i].plot(epochs_ndf, validation_loss_cv_NDF.loc[i, :], "-", color="C1")
        axes[i].set_title("CV fold: %i" %(i + 1))
        axes[i].set_xlabel("epoch")
    plt.show()
    fig.savefig("loss_comparison.png", bbox_inches='tight')

    # Accuracy
    std_acc_CNN = np.std(accuracy_cv_CNN)
    std_acc_NDF = np.std(accuracy_cv_NDF)

    fig, axes = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(10, 7))
    fig.text(0.07, 0.5, 'Accuracy', va='center', rotation='vertical')
    fig.suptitle("Accuracy comparison", fontsize=20)
    axes[0].set_ylim(0, 1)
    axes[0].set_title('Cross Validation(mean)')
    axes[0].bar("CNN", accuracy_cv_CNN.mean(), yerr=std_acc_CNN)
    axes[0].bar("CNN_NDF", accuracy_cv_NDF.mean(), yerr=std_acc_NDF)
    axes[1].set_title('Testset')
    axes[1].bar("CNN", test_results_CNN["accuracy"])
    axes[1].text("CNN", test_results_CNN["accuracy"] + 0.03,
                  str(round(test_results_CNN["accuracy"][0] * 100, 1)) + "%", fontsize='x-large')
    axes[1].bar("CNN_NDF", test_results_NDF["accuracy"])
    axes[1].text("CNN_NDF", test_results_NDF["accuracy"] + 0.03,
                  str(round(test_results_NDF["accuracy"][0] * 100, 1)) + "%", fontsize='x-large')
    plt.show()
    fig.savefig("accuracy.png", bbox_inches='tight')

    # AUC ROC CURVE
    if opt.n_class == 2:
        std_auc_CNN = np.std(auc_cv_CNN)
        std_auc_NDF = np.std(auc_cv_NDF)

        fig, axes = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(10, 7))
        fig.text(0.07, 0.5, 'Area under ROC Curve', va='center', rotation='vertical')
        fig.suptitle("AUC comparison", fontsize=20)
        axes[0].set_ylim(0, 1)
        axes[0].set_title('Cross Validation(mean)')
        axes[0].bar("CNN", auc_cv_CNN.mean(), yerr=std_auc_CNN)
        axes[0].bar("CNN_NDF", auc_cv_NDF.mean(), yerr=std_auc_NDF)
        axes[1].set_title('Testset')
        axes[1].bar("CNN", test_results_CNN["auc"])
        axes[1].text("CNN", test_results_CNN["auc"] + 0.03,
                      str(round(test_results_CNN["auc"][0] * 100, 1)) + "%", fontsize='x-large')
        axes[1].bar("CNN_NDF", test_results_NDF["auc"])
        axes[1].text("CNN_NDF", test_results_NDF["auc"] + 0.03,
                      str(round(test_results_NDF["auc"][0] * 100, 1)) + "%", fontsize='x-large')
        plt.show()
        fig.savefig("auc.png", bbox_inches='tight')

        # ROC CURVE
        fpr_CNN = roc_CNN.iloc[0, :]
        tpr_CNN = roc_CNN.iloc[1, :]

        fpr_NDF = roc_NDF.iloc[0, :]
        tpr_NDF = roc_NDF.iloc[1, :]

        plt.title("ROC Curve")
        plt.plot(fpr_CNN, tpr_CNN, label="CNN_only")
        plt.plot(fpr_NDF, tpr_NDF, label="CNN_NDF")
        plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), color="grey")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.show()

    # Plot the training_loss
    plt.title("Training loss (full trainingset)")
    plt.plot(epochs_cnn, training_loss_full_CNN.loc[0, :], label="CNN")
    plt.plot(epochs_ndf, training_loss_full_NDF.loc[0, :], label="CNN_NDF")
    plt.xlabel("epoch")
    plt.ylabel("loss value")
    plt.legend()
    plt.savefig("Training_loss_full.png", bbox_inches='tight')
    plt.show ()


if __name__ == '__main__':
    main()
