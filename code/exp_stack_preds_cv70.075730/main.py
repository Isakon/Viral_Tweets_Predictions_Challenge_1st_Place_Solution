import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import os, shutil, time
from tqdm import tqdm

from datasets import split2folds_viral_only
from models import get_model_class

osj = os.path.join; osl = os.listdir

#  ========= Constants ===================
soft_preds = True  # False  # soft (probs) or hard (from 1 to 5)
stack_name = "a_stack_eps140_alpha0001_replace_cv6862DNN_and_seeds1234_vs_cv6844_of_PL6992"  # _alph01"
# "a_stack_alph1e3_many_models_eps300_w_2layers_layer1_mult3"  # "a_stack_logreg"  #  "a_stack_mlp_3layers_soft_preds_9lgb_1catboost_models_jun23"
stack_model_name = 'stack_mlp_2layers'  # 'stack_mlp_3layers'  # 'stack_linear'  # 'stack_mlp_2layers'  # 'stack_mlp_3layers' # 'stack_mlp_1layer'  #  'stack_logistic'  # 'stack_mlp'
layer_one_multiplier = 3  # 4  # 10  # 5
params_model = dict(
    random_state = 24,
    max_iter = 140,  # 200,  # default = 200
    # learning_rate_init = 0.001, # default=0.001
    # learning_rate = 'invscaling',  # {'constant', 'invscaling', 'adaptive'}, default='constant'
    # hidden_layer_sizes = None,
    activation = 'logistic', # '{'identity', 'logistic', 'tanh', 'relu'}
    alpha = 0.0001,  #0.0001, # default=0.0001  L2 penalty (regularization term) parameter
    # batch_size = 60, # "auto", batch_size=min(200, n_samples)
    # tol = 1e-5,
    # early_stopping = 200  # bool, default=False
    # n_iter_no_change = 200, # default = 10
    # validation_fractionfloat, default=0.1 . Only used if early_stopping is True
    # verbose = False
    )
base_dir = '../../data/raw'
# used only 'tweet_id' and 'virality' columns
preprocessed_train_path = '../../data/processed/train_ids_virality_fold.csv'
preds_dir_paths = [
    '../../submissions/exp_cv690750_PL68.90189_catboost_Bootstrap_type_Bernoulli',
    '../../submissions/exp_cv688282__lgb_fidrop300',
    '../../submissions/exp_cv688180_lgb_refactored_jun23',
    '../../submissions/exp_cv688349__lgb_param_tune_jun27',
    # '../../submissions/exp0_run_first_for_feature_importances_PL68.93564_cv688789_lgb',
    '../../submissions/exp_cv686253_dnn0_seed0_improve_cv684461_BranchedDNN',
    '../../submissions/exp_dnn1_cv684394__seed1_cv6862_m_branched_dnn',
    '../../submissions/exp_dnn2_cv686389__seed2_cv6862_m_branched_dnn',
    '../../submissions/exp_dnn3_cv686625__seed3_cv6862_m_branched_dnn',
    '../../submissions/exp_dnn4_cv686355__seed4_cv6862_m_branched_dnn',
    ]

def logprint(log_str, add_new_line=True):
    # os.makedirs(out_dir, exist_ok=True)
    if add_new_line:
        log_str += '\n'
    print(log_str)
    with open(os.path.join(out_dir, f'log.txt'), 'a') as appender:
        appender.write(log_str + '\n')

def create_out_dir_from_paths(paths: list, s: str):
    dir_name = 'submission_stack_final'  # s+ '_' + '_'.join([os.path.basename(p_)[:12] for p_ in paths])
    out_dir = osj('../../submissions', dir_name)
    os.makedirs(out_dir)
    return out_dir

def select_model(model, stack_model_name, preds_dir_paths,
                     params_model, layer_one_multiplier):
    if stack_model_name == 'stack_mlp_1layer':
        model = model(hidden_layer_sizes=(len(preds_dir_paths) * 5) // 5,
                      **params_model)  # (1,)  # (len(preds_dir_paths)*5), 1)
    elif stack_model_name == 'stack_mlp_2layers':
        model = model(
            hidden_layer_sizes=(len(preds_dir_paths) * layer_one_multiplier, len(preds_dir_paths) * 5 // 5,),
            **params_model)  # (1,)  # (len(preds_dir_paths)*5), 1)
    elif stack_model_name == 'stack_mlp_3layers':
        model = model(hidden_layer_sizes=(len(preds_dir_paths) * 5, len(preds_dir_paths) * 5 // 2,
                                          len(preds_dir_paths) * 5 // 5),
                      **params_model)  # (1,)  # (len(preds_dir_paths)*5), 1)
    elif stack_model_name == 'stack_linear':
        model = model
    else:
        raise NotImplementedError
    return model


def get_xy_train(train, train_stack):
    if soft_preds:
        x_train = np.concatenate(train_stack, axis=-1)
    else:
        x_train = np.concatenate(train_stack, axis=-1)
    y_train = train['virality'].values
    return x_train, y_train

# def get_folds_local(train, n_folds, folds_split_path='../../data/processed/train_ids_virality_fold.csv')
#     # '../../data/processed/folds_split/train_folds_split_folds_num_folds5_seed_folds24_type_split_user_viral_datetime_21_06_time_20_58/train_folds_split_folds_num_folds5_seed_folds24_type_split_user_viral_datetime_21_06_time_20_58.csv'):
#     if n_folds==5:
#         folds = pd.read_csv(folds_split_path)
#         if 'fold' in train.columns:
#             del train['fold']
#         train = train.merge(folds['tweet_id','], how='left', on='tweet_id').reset_index(drop=True)
#     else:
#     train = split2folds_viral_only(train, n_folds, seed_folds=24, label_col='virality')  # label_cols=['tweet_user_id', 'virality'])
#     return train

def evaluate_stack(model, train, x_all, y_all, x_test, n_folds=5):
    accuracies = []
    preds_val = np.zeros((len(x_all), 5), dtype=np.float32)
    preds_test = np.zeros((len(x_test), 5), dtype=np.float32)
    # train = get_folds_local(train,n_folds)
    for fold in tqdm(range(n_folds)):
        train_idx = train[train['fold'] != fold].index
        val_idx = train[train['fold'] == fold].index
        x_train = x_all[train_idx]
        y_train = y_all[train_idx]
        x_valid = x_all[val_idx]
        y_valid = y_all[val_idx]

        model.fit(x_train, y_train)
        preds_val[val_idx] = model.predict_proba(x_valid)
        preds_test += model.predict_proba(x_test)/ n_folds
        accuracy_fold = 100*accuracy_score(y_valid, np.argmax(preds_val[val_idx], axis=1))
        accuracies.append(round(accuracy_fold,10))
        logprint(f"Fold {fold}, accuracy = {accuracy_fold}")
    ave_accuracy = round(np.mean(accuracies),5)
    oof_accuracy = round(accuracy_score(y_all, np.argmax(preds_val, axis=1)) * 100, 5)
    logprint(f"average_accuracy = {ave_accuracy:5f}")
    logprint(f"oof_accuracy = {ave_accuracy}")
    logprint(f"accuracies:\n {accuracies}")
    np.savetxt(osj(out_dir, f"accuracy_scores_ave{ave_accuracy}_oof{oof_accuracy}.txt"), np.array(accuracies))
    return ave_accuracy, oof_accuracy, preds_test


def main():
    for p_ in preds_dir_paths:
        assert os.path.isdir(p_), f"path wrong : {p_}"

    sub = pd.read_csv(osj(base_dir, 'solution_format.csv'))
    sub_ave = sub.copy()
    train = pd.read_csv(preprocessed_train_path, usecols=['tweet_id','virality', 'fold'])
    train_stack, test_stack = [], []
    for i, dirpath in enumerate(preds_dir_paths):
        preds_valid = pd.read_csv(osj(dirpath, 'preds_valid.csv'))
        if i==0: len_preds_valid0 = len(preds_valid)
        preds_valid = preds_valid.iloc[:len(train)]
        assert preds_valid['tweet_id'].equals(train['tweet_id']), f"preds_valid['tweet_id'].equals(train['tweet_id']) = {preds_valid['tweet_id'].equals(train['tweet_id'])}"
        if soft_preds:
            preds_valid = preds_valid.iloc[:, 1:].values
        else:
            preds_valid = np.argmax(preds_valid.iloc[:,1:].values, axis=1)[:,np.newaxis]
        train_stack.append(preds_valid)

        fn_preds_test = 'preds_test_soft.csv' if soft_preds else 'preds_test.csv'
        preds_test = pd.read_csv(osj(dirpath, fn_preds_test))
        preds_test = preds_test.iloc[:len(train)]
        # some preds_test have no 'tweet_id' columns
        if 'tweet_id' in preds_test:
            iloc_start = 1
        else:
            iloc_start = 0
        assert 5 <= preds_test.iloc[:, iloc_start:].shape[1] <= 6, "preds_test folds must be = 5"
        if soft_preds:
            preds_test = preds_test.iloc[:, iloc_start:].values
        else:
            preds_test = (preds_test.iloc[:, iloc_start:]
                              .mode(axis=1).iloc[:, 0].values[:,np.newaxis])
        test_stack.append(preds_test)

    model = get_model_class(stack_model_name)
    model = select_model(model, stack_model_name, preds_dir_paths,
                     params_model, layer_one_multiplier)
    print(f"model: {model}")
    #x_train = np.hstack([x[i][:, np.newaxis] for x in train_stack])
    if soft_preds:
        x_train = np.concatenate(train_stack, axis=-1)
    else:
        x_train = np.concatenate(train_stack, axis=-1)
    y_train = (train['virality'].values - 1).astype(int)
    assert set(np.unique(y_train)) == set(range(5))  #  {0, 1, 2, 3, 4}
    x_test = np.concatenate(test_stack, axis = 1)
    ave_accuracy, oof_accuracy, preds_test_ave = evaluate_stack(model, train, x_train, y_train, x_test)
    logprint(f"ave_accuracy = {ave_accuracy:.5f}")
    # retrain on full datasets
    model = get_model_class(stack_model_name)
    model = select_model(model, stack_model_name, preds_dir_paths,
                         params_model, layer_one_multiplier)
    model.fit(x_train, y_train)
    sub_ave = sub_ave.iloc[:len(preds_test_ave)]
    sub = sub.iloc[:len(preds_test_ave)]
    sub_ave.loc[:, 'virality'] = 1 + np.argmax(preds_test_ave, axis=1)
    assert len(np.setdiff1d(sub_ave['virality'].unique(), np.arange(1, 6))) == 0
    # sub_ave.to_csv(osj(out_dir, 'submission_average_folds.csv'), index=False)
    # print(f"sub_ave.shape: {sub_ave.shape}")

    sub.loc[:, 'virality'] = 1 + model.predict(x_test).astype(int)
    assert len(np.setdiff1d(sub['virality'].unique(), np.arange(1,6)))==0
    sub.to_csv(osj(out_dir, 'submission.csv'), index=False)
    logprint(f"sub.shape: {sub.shape}")
    train_vc = (train['virality'].value_counts() / train.shape[0]).round(4)
    sub_vc = (sub['virality'].value_counts() / sub.shape[0]).round(4)
    vc_ = pd.concat([sub_vc, train_vc], axis=1)
    vc_.columns = ['sub', 'train']  # , 'accuracy']
    logprint(f"sub and train ['virality'].value_counts():\n{vc_.to_string()}")
    logprint(f"sub.head():\n{sub.head().to_string()}")
    # new_outdir_name = osj(os.path.dirname(out_dir), f"cv{oof_accuracy:5f}_{ave_accuracy:5f}_"+os.path.basename(out_dir))
    # print(f"new out_dir:\n{new_outdir_name}")
    # shutil.move(out_dir, new_outdir_name)


if __name__ == '__main__':
    out_dir = create_out_dir_from_paths(preds_dir_paths, stack_name)
    print(f"============\nStarting...\nout_dir: {out_dir}")
    shutil.copy(__file__, out_dir)
    main()


