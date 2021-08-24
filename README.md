## Viral-Tweets-Prediction-Challenge-1st-Place-Solution

### Context
The aim of the challenge is to predict the level of virality of a tweet using the tweet and the user data including image and text extracted features.  
The metrics is the average accuracy for all 5 classes of virality.  
The link to the description page: https://bitgrit.net/competition/12#  

### Solution Overview

- Training data is split into 5 folds based on joint ['tweet_user_id','virality'] columns so that the users and virality are distributed evenly between folds.

- Models used are 3 lightgbm models, 1 catboost and 5 different random seeds for 1 neural network architecture with pytorch.

- Neural network architecture consists of 2 branches of dense layers - one for tabular numerical and categorical features and the second for extracted text and image features which are concatenated at the second hidden layer and then outputs 5 class predictions. Loss function is the crossentropy with different class weights.

- Hyperparameters of base models are in config.yaml files of respective subdirectories in 'code' directory

- All base model scripts save 'preds_valid.csv' and 'preds_test_soft.csv' which are used in stacking. They each consist of 5 columns of probability predictions for each class.

- Stacking is done with sklearn MLPClassifier, 2 layers, applied to 45 columns of probability predictions (5 classes * 9 base models)

### Environment
OS: Linux Ubuntu 18.04  
Python 3.7.10  
CUDA Version:  10.2  
nvidia drivers:  460.80  
Package manager: conda 4.10.1  
To recreate environment run: 
$conda env create -f environment.yml

### Hardware
RAM: 16 GB + swap space 16 GB  
Disk space > 50GB  
CPU: Intel i5 @ 2.90GHz - x86_64, 6 cores  
GPU: RTX2070

### Directory structure
├── code

│   ├── exp0_run_first_for_feature_importances_PL68.93564_cv688789_lgb  
│   ........├──────── config.yaml  
│   ........└──────── ...  
│   ........└──────── main.py  
│  
│   └── exp_cv690750_PL68.90189_catboost_Bootstrap_type_Bernoulli  
│   ........├──────── config.yaml  
│   ........└──────── ...  
│   ........└──────── main.py  │  

│   └── exp_dnn0_cv686253__improve_cv684461_BranchedDNN  
│   ........├──────── config.yaml  
│   ........└──────── ...  
│   ........└──────── main.py  

│   └── exp_stack_preds_cv70.075730  
│   ........├──────── config.yaml  
│   ........└──────── ...  
│   ........└──────── main.py  
│  
│   └── train_group.sh file

├── data

│   ├── processed

│   └── raw

└── submissions

└── README.md

└── environment.yaml



 ### Files created and used during execution
- 'data/processed/feature_importances_lgb_cv68879.csv' - feature importances of lightgbm model, created during execution of 'code/exp0_run_first_for_feature_importances_PL68.93564_cv688789_lgb/lgbm_train.py' and used to drop features in some base models
- 'data/processed/train_ids_virality_fold.csv' - created during preprocessing of 'code/exp_cv688282__lgb_fidrop300/lgbm_train.py' base model script and used for stacking
- 
 ### Data
 - Save the official dataset to location: 'data/raw' subdirectory
 
 ### Instructions to Run training and predictions
 - Create environment 'viral_tweets' with  
    $conda env create -f environment.yml  
    $conda activate viral_tweets

 - Change to 'code' directory:  
        $cd code
 - To debug, run:  
        $bash train_group.sh -d True  
        This will run training and prediction scripts on sample datasets (2000 samples)
 - To run training and prediction on full dataset, run:  
        $bash train_group.sh -d False
   - To run any of the base models training and prediction, go to its subdirectory (e.g. 'code/exp_cv688349__lgb_param_tune_jun27') and run  
        $python main.py
 - Before each run, the output subdirectories in the 'submissions' folder should be deleted or moved.  
 - In debug mode total runtime is around 20 min., in full mode approximately 9-10 hours


### Explanation of the Pipeline

'train_group.sh' runs 4 lightgbm models (1 model is used only to output feature importances), 1 catboost, and 5 random seed versions of 1 neural network (pytorch). They each output 'preds_valid.csv' and 'preds_test_soft.csv' with 5 soft class prediction columns (probabilities) in respective subdirectories of 'submissions' folder.

Then it runs 'main.py' at 'exp_stack_preds_cv70.075730' to stack the models predictions which fetches 'preds_valid.csv' predicitons from those subdirectories and trains and evaluates 5-fold stacking model with sklearn.MLPClassifier printing out stacked models accuracy score. Final stacking model is trained with the same parameters on all samples of stacked 'preds_valid.csv' data. 

The trained model runs prediction on stacked 'preds_test_soft.csv' base models data and saves submissions.csv file in 'submissions/stacked_.... ' subdirectory which serves as the final submision file.

Configuration files (config.yaml) for each base model are in the respective subdirectory of 'code' directory. Configuration parameters 'debug', 'seed_folds', 'seed_other' in the 'config.yaml' files can be changed/overwritten by command line argument.

The main.py from subdirectory 'code/exp0_run_first_for_feature_importances_PL68.93564_cv688789_lgb' saves feature importances to ../../data/processed/feature_importances_lgb_cv68879.csv which are used by some other base models to drop features

The main.py from subdirectory 'code/exp_cv688282__lgb_fidrop300' saves train[['tweet_id','tweet_user_id','virality']] with duplicates from training set dropped to '../../data/processed/train_ids_virality_fold.csv' which are used for stacking

### Which data files are being used?
- All data files are used except some base models do not use user description data ('user_vectorized_descriptions.csv'). The option is set with 'drop_user_des_feats' option in 'config.yaml' files in each base models respective subdirectory.

### How are these files processed?
- Data prepocessing is conducted in 'get_data_stage1' and 'get_data_stage2' functions of 'Features' class in 'datasets.py' module of each base model subdirectory. Slightly different features are created in each base model subdirectory. But main pipeline is the following:
   1. 'train_tweets_vectorized_media.csv' are grouped by 'tweet_id' and the means are merged with 'train_tweets.csv'. The feature 'num_media' with the number of media samples for each tweet_id and 'tweet_has_media' (if the tweet_id has media image) are created. 
   2. after merging text, user image and user description files with the main feature dataframe, dates related features are created reflecting the number of tweets created by user at year, month, hour, and different stats columns for 'tweet_hashtag_count',  'tweet_url_count',  'tweet_mention_count' for each calendar quarter for the period.
   3. Ratio features like 'user_followers_count'/'days_since_tweet', 'tweet_mention_count'/'user_followers_count' are created
   4. Some count features are binned in 'def bin_feats' function
   5. topic_id ids are either one-hot-encoded or the ids at position are used as features
   6. 'user_virality' feature, average 'virality' for the user in the training set, is added for the training set and then extended to test set by matching to the corresponding tweet_user_id.
   7. frequency columns are generated by 'freq_encoding' function
   8. 'tweet_language_id' column is one-hot-encoded and only languages 0,1,3 are used.
   9. in some base models logarithms of count features are used
   
   - 47 duplicate rows are dropped from training set
   - The data is standardized by sklearn StandardScaler
   - The preprocessed train and test dataframes are saved and then loaded to reproduce the original results (the save files are deleted immediately to save disk space)

### Final submission.csv location
- 'train_group.sh' will save 'submission.csv' in 'submissions/submission_stack_final' directory





