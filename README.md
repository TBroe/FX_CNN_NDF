# FX_CNN_NDF - README
Tobias Broeckl, 07.02.2019

## Goal of the project
Goal is to test two approaches in the field of Machine Learning on FX data for short-term price 
prediction and potential practical application. The two approaches are CNN FX regression (CNN FX), introduced in 2017, 
and the Neural Decision Forest (NDF), introduced in 2015.  The contributions of this thesis to the field of FX 
related Machine Learning are the following:

– Adjustment of the CNN FX regression approach for classification.

– Extension of the CNN FX regression approach with the NDF classification model.

– Application of the NDF approach on time series data.

## Directory overview

    ./data 
   This directory needs to be created manually. It stores the raw currency timeseries data.
   Requirements: Raw currency data .txt file to be downloaded from http://www.forextester.com/data/datasources. 
   Manual transformation from .txt to .csv file necessary.

    ./python code 
   Stores the python scripts used for data preparation, training, testing and evaluation

    ./jupyter notebook code
   Stores the jupyter notebook scripts to get further insights on data processing and model evaluation.
     
    ./results
   This directory will be created automatically upon completing step 2 below. 
   It stores the trained model and results from both the training and testing python scripts.

    ./results/figures 
   This directory will be created automatically upon completing step 2 below.
   Stores the generated figures from the results script.


## Steps to run the code

Note:
Prefix for all python script references in this README is "FX_CNN_NDF". 
Currently only the two currency pairs: EUR/USD and GBP/JPY have been implemented as currency choices in the model.
   
   ####Step 1: 
   Run script .data_preprocessing.py to generate the .csv files that are later used as training and test input. 
   
   Note: The generated .csv files will be stored in the ./data directory. 
   
   For more details on the pre-processing please see the jupyter Notebook "FX_CNN_data_Day_Hour.ipynb".
   The Notebook is stored in the ./code/jupyter directory.
   
   ####Step 2: 
   Run .train.py script to generate the trained model and training results based on the pre-processed inputs.
   
   All training results are stored in the ./results directory.  
   
   Several training options are possible and have to be passed to the script via the parser. 
   
   Please see "Parser arguments" below for further info.
   
   The parsed arguments used for training are saved in arguments.txt.
     
   After training is completed the final model parameters are stored in "trained_model" file.
   
   Note: The main() function in train.py script calls functions from both .model.py and .data.py scripts.
   
   ####Step 3: 
   Run the .test.py script to evaluate trained model on the testset.
   
   The testset is generated in the with the prep_data() function in .data.py
   
   To run the test.py make sure to parse the same arguments as were used to generate the trained model. 
   
   The testing results are printed directly in the console and are stored as a .csv file in the ./results directory.
   
   ####Step 4: 
   Inspect and compare the training and testing result using the .results.py script. 
   
   This script imports the earlier generated results of the train and test script.
   
   The output of the .results.py script include the comparisons of accuracy and test/validation loss across all CV-folds. 
   
   Note: If binary classification was used in training, the AUC evaluation is also compared. 
   
   The resulting plots are saved for later inspection in the ./results/figures directory.
    
##Parser arguments
   
   ### General
   
    -CNN_only 
   If this arg is passed then only the CNN is trained. If not passed: A CNN with an appended NDF is trained.
   
    -daily_forecast 
   If this arg passed then day/hour setup is used. If not passed: Hour/minute setting can be used (TBD)
   
    -currency 
   Choice between EUR/USD and JPY/GBP.
   
    -n_class 
   Choice between two and four classes.
   
    -class_method  
   Classification method used. More details in "FX_CNN_data_Day_Hour.ipynb". Default is method 2.
   
    -periods_ahead 
   Specifies the prediction horizon.
   
    -k 
   Number of folds used in k-fold Cross-Validation. Default is 5 folds.
   
    -split 
   Split used to divide the data into training and testset. Default value is 0.75.
   
    -epochs 
   Number of epoch to be used for training. Epochs larger than 50 is advised.
   
    -batch_size 
   Number of samples used in a single forward / backward pass of the model.
   
    -dropout_rate 
   Percentage used for the dropout layers in the CNN. Default value is 0.3.
   
   ### NDF specific
    -n_tree 
   Number of trees used in the Neural Decision Forest.
   
    -tree_depth 
   Number of node levels of an individual tree in the NDF.
   
    -tree_feature_rate 
   Defines which random subset of features is used for a single tree in the forest.
   
   ### Examples
   
   Example 1: 
   
    FX_CNN_NDF_train.py -daily_forecast -CNN_only -n_class 2 -epochs 75 -k 5 -batch_size 10 -periods_ahead 1
    
   CPU training of a standalone CNN in the day/hour setting with binary classification
   and one day-ahead prediction for 75 epochs.
   
   Example 2: 
   
    FX_CNN_NDF_train.py  -daily_forecast -n_class 4 -epochs 150 -k 5 -batch_size 10 -periods_ahead 1 -n_tree 20 -tree_depth 5
                                      
   CPU training of a standalone CNN in the day/hour setting with binary classification 
   and one day-ahead prediction for 75 epochs.

##Packages

    torch           1.0.1
    torchvision     0.2.1
    matplotlib      3.0.2
    numpy           1.16.1	
    pandas          0.24.1	
    scikit-learn    0.20.2	
    
           
  
 
  
    
    
