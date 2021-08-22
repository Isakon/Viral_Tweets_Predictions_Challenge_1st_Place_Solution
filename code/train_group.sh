#!/bin/sh
# run  bash train_group.sh -d True for debug mode
# or bash train_group.sh -d False to run full
while getopts d: flag
do
  case "${flag}" in
    d) DEBUG=${OPTARG}
  esac
done


echo "DEBUG: " $DEBUG
cd exp_cv690750_PL68.90189_catboost_Bootstrap_type_Bernoulli/
python main.py  --debug $DEBUG

# this saves feature_importances.csv file in ../data/processed dir
cd ../exp0_run_first_for_feature_importances_PL68.93564_cv688789_lgb
python main.py --debug $DEBUG

cd ../exp_cv688180_lgb_refactored_jun23
python main.py --debug $DEBUG

# this saves train[['tweet_id','tweet_user_id','virality']] for us in exp_stack_preds_cv70.075730
cd ../exp_cv688282__lgb_fidrop300
python main.py --debug $DEBUG

cd ../exp_cv688349__lgb_param_tune_jun27
python main.py --debug $DEBUG

cd ../exp_dnn0_cv686253__improve_cv684461_BranchedDNN
python main.py --debug $DEBUG --seed 24

cd ../exp_dnn1_cv684394__seed1_cv6862_m_branched_dnn
python main.py --debug $DEBUG --seed 1

cd ../exp_dnn2_cv686389__seed2_cv6862_m_branched_dnn
python main.py --debug $DEBUG --seed 2

cd ../exp_dnn3_cv686625__seed3_cv6862_m_branched_dnn
python main.py --debug $DEBUG --seed 3

cd ../exp_dnn4_cv686355__seed4_cv6862_m_branched_dnn
python main.py --debug $DEBUG --seed 4

cd ../exp_stack_preds_cv70.075730
python main.py


