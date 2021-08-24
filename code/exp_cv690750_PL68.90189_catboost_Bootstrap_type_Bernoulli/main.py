import os, glob, gc, time, yaml, shutil, random
import addict
import argparse
from collections import defaultdict
from tqdm import tqdm
import numpy as np 
import pandas as pd 

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, QuantileTransformer, KBinsDiscretizer
from datasets import (Features, transform_joint, normalize_npnan,
                      NormalizeFeats, get_feat_cols, get_folds, split2folds_user_viral,
                        save_preprocessed
                      )

from catboost import CatBoostClassifier, Pool

import warnings
warnings.filterwarnings('ignore')
pd.set_option("display.max_colwidth", 100)
pd.set_option("display.max_rows", 20)
np.set_printoptions(precision=4)

osj = os.path.join; osl = os.listdir

def read_yaml(config_path='./config.yaml'):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return config

def parse_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--kernel_type', type=str, required=True)
    parser.add_argument('--debug', type=str,  default="False")
    parser.add_argument('--seed', type=int, default=24)
    args, _ = parser.parse_known_args()
    return args

def gettime(t0):
    """return a string of time passed since t0 in min.
    Ensure no spaces inside (for using as the name in files and dirs"""
    hours = int((time.time() - t0) / 60 // 60)
    mins = int((time.time() - t0) / 60 % 60)
    return f"{hours:d}h{mins:d}min"


def logprint(log_str, add_new_line=True):
    # os.makedirs(out_dir, exist_ok=True)
    if add_new_line:
        log_str += '\n'
    print(log_str)
    with open(os.path.join(out_dir, f'log.txt'), 'a') as appender:
        appender.write(log_str + '\n')

def copy_code(out_dir: str, src_dir='./'):
    code_dir = os.path.join(out_dir, 'code')
    os.makedirs(code_dir, exist_ok=False)
    py_fns = glob.glob(os.path.join(src_dir, '*.py'))
    py_fns += glob.glob(os.path.join(src_dir, '*.yaml'))
    for fn in py_fns:
        shutil.copy(fn, code_dir)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # for faster training, but not deterministic

def compare_data(test,
                 compare_path='/home/isakev/challenges/viral_tweets/data/processed/test_data_lgb_NoImpute_NoOhe_jun23.csv'):
    print(f"preprocessed path:\n{compare_path}")
    # print("cfg.test_preprocessed_path\n:", cfg.test_preprocessed_path)
    # assert os.path.basename(compare_path) == os.path.basename(cfg.test_preprocessed_path)

    test_compare = pd.read_csv(compare_path)

    cols_here_not_in_preproc = set(test.columns).difference(set(test_compare.columns))
    cols_preproc_not_in_here = set(test_compare.columns).difference(set(test.columns))

    print(f"cols_preproc_not_in_here:\n{cols_preproc_not_in_here}")
    print(f"cols_here_not_in_preproc:\n{cols_here_not_in_preproc}")

    print(f"test.isnull().sum().sort_values().tail():\n{test.isnull().sum().sort_values().tail()}")
    print(f"\ntest_compare.isnull().sum().sort_values().tail():\n{test_compare.isnull().sum().sort_values().tail()}")
    # print()
    minus_ones_compare, minus_ones_here = [], []
    for col in test_compare.columns:
        minus_ones_compare.append((test_compare[col] == -1).sum())
        minus_ones_here.append((test_compare[col] == -1).sum())
    print(f"minus_ones_compare:{sum(minus_ones_compare)}")
    print(f"minus_ones_here:{sum(minus_ones_here)}")

    assert len(cols_preproc_not_in_here) == 0
    assert len(cols_preproc_not_in_here) == 0
    if len(test) > 5000:
        assert len(test) == len(test_compare), f"len test = {len(test)} , len test_compare = {len(test_compare)}"
    assert sum(minus_ones_compare) == sum(minus_ones_here)
    min_len = min(len(test), len(test_compare))
    test = test.iloc[:min_len].reset_index(drop=True)
    test_compare = test_compare[:min_len]
    unequals = test.compare(test_compare)
    print(f"test.compare(test_compare).shape[1] = {unequals.shape[1]}")
    print(f"test.compare(test_compare).columns: {unequals.columns}")

    diffs_ls = []
    for col0 in unequals.columns.get_level_values(0):
        diffs = unequals[(col0, 'self')] - unequals[(col0, 'other')]
        diffs_ls.append(np.sum(diffs) / len(diffs))

    argsorted_cols = unequals.columns.get_level_values(0)[np.argsort(diffs_ls)]

    print(f"np.sum(diffs_ls = {np.sum(diffs_ls)}")
    cols_diff_ = [(col, diff_) for (col, diff_) in zip(argsorted_cols[-10:], np.sort(diffs_ls)[-10:])]
    print(f"some  diffs_ls[-10:]:\n{cols_diff_}")

    # assert test.compare(test_compare).shape[1] == 0, "test.compare(test_compare).shape[1] == 0"


def create_out_dir(experiment_name, model_arch_name, n_folds, folds_to_train, debug):
    # datetime_str = time.strftime("%d_%m_time_%H_%M", time.localtime())
    # folds_str = '_'.join([str(fold) for fold in folds_to_train])
    # out_dir = '../../submissions/{}_m_{}_ep{}_bs{}_nf{}_t_{}'.format(
    #             experiment_name, model_arch_name, cfg.n_epochs, cfg.batch_size, n_folds,  datetime_str)   # bs, weight_decay, , folds_str,
    # if debug:
    #     out_dir = osj(os.path.dirname(out_dir), 'debug_' + os.path.basename(out_dir))
    out_dir = cfg.out_dir
    models_outdir = osj(out_dir, 'models')
    os.makedirs(out_dir)
    os.makedirs(models_outdir)
    return out_dir, models_outdir


def save_model(path, model, epoch, best_score, save_weights_only=False):
    if save_weights_only:
        state_dict = {
            'model': model.state_dict(),
            'epoch': epoch,
            'best_score': best_score,
            }
    else:
        scheduler_state = scheduler.state_dict() if scheduler else None
        state_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler_state,
            'epoch': epoch,
            'best_score': best_score,
        }
    torch.save(state_dict, path)

def rename_outdir_w_metric(out_dir, ave_metric, ave_epoch):
    # renames out_dir - prefix_ = f"cv{ave_metric_str}_"
    # after cross-validation:
    # rename out_dir adding ave_epoch_cv(metric) to the name
    if ave_metric < 1:
        ave_metric_str = f"{ave_metric:.6f}"[2:]
    elif ave_metric < 1000:
        ave_metric_str = f"{ave_metric:.5f}".replace('.', '_')
    else:
        ave_metric_str = f"{int(ave_metric)}"
    if ave_epoch: ave_epoch = int(ave_epoch)

    prefix_ = f"cv{ave_metric_str}_"
    suffix_ = f"_e{ave_epoch}_cv{ave_metric_str}"
    new_base_name = prefix_ + os.path.basename(out_dir) + suffix_
    out_dir_new_name = osj(os.path.dirname(out_dir), new_base_name)
    # os.rename(out_dir, out_dir_new_name)
    assert not os.path.exists(out_dir_new_name), f"\nCan't rename: the path exists ({out_dir_new_name}"
    print(f"new out_dir directory name:\n{os.path.basename(out_dir_new_name)}")
    shutil.move(out_dir, out_dir_new_name)

    return out_dir_new_name


def get_cols2normalize(feat_cols):
    cols2normalize = [col for col in feat_cols if (not col.startswith('img_feature_'))
                                                       and (not col.startswith('feature_'))
                                                        and (not col.startswith('user_des_feature_'))
                                                        and (not col.startswith('user_img_feature_'))
                              ]
    return cols2normalize


def drop_duplicates_func(train, feat_cols):
    n = train.shape[0]
    train.drop_duplicates(subset=feat_cols, inplace=True)
    train.reset_index(drop=True, inplace=True)
    # [~train.duplicated(subset=feat_cols)].reset_index(drop=True)
    print(f"Dropped {n - train.shape[0]} duplicated rows from train. train.shape = {train.shape}")
    return train

def run(fold, train, test, feats, categorical_columns):  # cat_idxs):

    def feats2list(df, feats, categorical_columns):
        # separate categorical and non-categorical features into lists and concatenate
        # pass new categorical indices
        noncat_cols = [col for col in feats if col not in categorical_columns]
        X_cat_list = df[categorical_columns].values.tolist()
        X_noncat_list = df[noncat_cols].values.tolist()
        X_list = [cat+noncat for (cat,noncat) in zip(X_cat_list,X_noncat_list)]
        cat_new_idxs = np.array(range(len(X_cat_list[0])))
        return X_list, cat_new_idxs

    t_start_fold = time.time()
    train_fold = train[train['fold'] != fold].copy()
    val_fold = train[train['fold'] == fold]
    # test_fold = test.copy()

    if isinstance(cfg.adversarial_drop_thresh, float) and cfg.adversarial_valid_path:
        # drop adversarial_valid samples from train
        # all fold 0(of 5) train set = 23662
        thresh = cfg.adversarial_drop_thresh
        # 0.23= = =cv68.1013
        # 0.24=23489=0.68425=cv68.1554
        # 0.25=23177=0.6830=cv67.9593
        # 0.27=21342=0.68636=cv68.0463
        adv_preds = pd.read_csv(cfg.adversarial_valid_path)
        drop_ids = adv_preds.loc[(adv_preds['is_test'] == 0) & (adv_preds['preds'] < thresh), 'tweet_id'].values
        print(f"Before adversarial cutoff, train_fold.shape = {train_fold.shape}")
        train_fold = train_fold[~train_fold['tweet_id'].isin(drop_ids)]

    # X_train = train_fold[feats].values
    # X_valid = val_fold[feats].values
    y_train = train_fold['virality'].values
    y_valid = val_fold['virality'].values
    # X_test = test[feats].copy().values

    X_train_list, cat_new_idxs = feats2list(train_fold, feats, categorical_columns)
    X_valid_list, _ = feats2list(val_fold, feats, categorical_columns)
    X_test_list, _ = feats2list(test, feats, categorical_columns)
    del train_fold, val_fold, test; _ = gc.collect()
    # baseline_value = y_train_first.mean()
    # train_baseline = np.array([baseline_value] * y_train_second.shape[0])
    # test_baseline = np.array([baseline_value] * y_test.shape[0])
    train_pool = Pool(X_train_list, y_train, cat_features=cat_new_idxs)
    valid_pool = Pool(X_valid_list, y_valid, cat_features=cat_new_idxs)
    test_pool = Pool(X_test_list, cat_features=cat_new_idxs)

    logprint(f"Train set n = {len(y_train)}, Valid set n = {len(y_valid)}, Num feats = {len(X_train_list[0])}")
    logprint(f"{time.ctime()} ===== Fold {fold} starting ======")
    del X_train_list, X_valid_list, X_test_list; _ = gc.collect()

    model_file = os.path.join(osj(models_outdir, f'model_best_fold{fold}.cbm'))
    stats = {}

    model = CatBoostClassifier(**cfg.catboost_params, cat_features=cat_new_idxs,
                               train_dir=out_dir)
    params_from_model = model._init_params

    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

    params_from_model = model.get_all_params()
    params_from_model = {k: v for (k, v) in params_from_model.items() if k not in ['cat_features']}
    # print(f"params from model:\n{params_from_model}")

    if fold == cfg.folds_to_train[0]:
        print(f"model params: {model}")

    print(f'Best Iteration: {model.best_iteration_}')
    # print(f"evals_result:\n{evals_result}")
    stats['best_iter'] = model.best_iteration_
    evals_res_df = pd.DataFrame({'train_accuracy': model.evals_result_['learn']['Accuracy'][::100],
                                'valid_accuracy': model.evals_result_['validation']['Accuracy'][::100]})
    evals_res_df.to_csv(osj(out_dir, f"evals_result_fold{fold}.csv"), index=False)

    # train score
    preds_train_fold = model.predict(train_pool)
    # train_rmse = np.sqrt(mean_squared_error(y_train, preds_train_fold))
    #acc_train = accuracy_score(y_train, np.argmax(preds_train_fold, axis=1))
    acc_train = accuracy_score(y_train, np.argmax(preds_train_fold, axis=1))


    # validation score
    preds_val_fold = model.predict_proba(valid_pool)
    acc_valid = accuracy_score(y_valid, np.argmax(preds_val_fold, axis=1))
    # y_pred_valid = rankdata(y_pred_valid) / len(y_pred_valid)

    # save model
    model.save_model(model_file, format="cbm")   # "json"

    # predict test
    preds_test_fold = model.predict_proba(test_pool)

    stats['acc_train'] = acc_train
    stats['acc_valid'] = acc_valid
    stats['fold'] = fold

    content = f'lr:{model.learning_rate_}, accuracy train: {acc_train:.5f}, accucary valid: {acc_valid:.5f}'
    print(content)
    print(f"From model.best_score_: {model.best_score_}")
    print(f"ACCURACY: {acc_valid: .5f} \tFold train duration = {gettime(t_start_fold)}\n\n {'-' * 30}")
    feature_imp_fold_df = pd.DataFrame({'feature': model.feature_names_,
                                        f'fold_{fold}': model.feature_importances_})

    return preds_val_fold, preds_test_fold, stats, feature_imp_fold_df


def main(out_dir, cfg):
    t_get_data = time.time()
    if cfg.load_train_test:
        print(f"Loading preprocessed train and test...Path train:\n{cfg.train_preprocessed_path}")
        train = pd.read_csv(cfg.train_preprocessed_path, nrows=n_samples)
        test = pd.read_csv(cfg.test_preprocessed_path, nrows=n_samples)
        if not cfg.add_user_virality:
            del train['user_virality'], test['user_virality']
            _ = gc.collect()
        (feat_cols, media_img_feat_cols, text_feat_cols,
         user_des_feat_cols, user_img_feat_cols, feats_some) = get_feat_cols(train)
        # train = drop_duplicates_func(train, feat_cols)
    else: # preprocess raw data
        # assert not os.path.exists(cfg.train_preprocessed_path), f"file exists: {cfg.train_preprocessed_path}"
        # assert not os.path.exists(cfg.test_preprocessed_path), f"file exists: {cfg.test_preprocessed_path}"
        features = Features()
        t_get_data = time.time()
        traintest = features.get_data_stage1(cfg, base_dir, n_samples=n_samples)
        train, test = features.get_data_stage2(cfg, traintest)

        (feat_cols, media_img_feat_cols, text_feat_cols,
         user_des_feat_cols, user_img_feat_cols, feats_some) = get_feat_cols(train)
        train = drop_duplicates_func(train, feat_cols)

        if cfg.save_then_load:
            # saving and loading preprocessed data in order to reproduce the score
            # if cfg.save_train_test:  # and (not cfg.debug):  # "../../data/preprocessed/"
            save_preprocessed(cfg, train, test, path_train=cfg.train_preprocessed_path,
                              path_test=cfg.test_preprocessed_path)
            train = pd.read_csv(cfg.train_preprocessed_path, nrows=n_samples)
            test = pd.read_csv(cfg.test_preprocessed_path, nrows=n_samples)
            # os.remove(cfg.train_preprocessed_path)
            # os.remove(cfg.train_preprocessed_path)

        # if cfg.save_train_test:  # and (not cfg.debug):  # "../../data/preprocessed/"
        #     save_preprocessed(cfg, train, test, path_train=cfg.train_preprocessed_path,
        #                       path_test=cfg.test_preprocessed_path)
        #     print(f"Saved train and test after initial preprocess at:\n{cfg.train_preprocessed_path}")

    train = get_folds(cfg, train)

    (feat_cols, media_img_feat_cols, text_feat_cols,
         user_des_feat_cols, user_img_feat_cols, feats_some) = get_feat_cols(train)

    if cfg.drop_tweet_user_id:
        train.drop('tweet_user_id', 1, inplace=True)
        test.drop('tweet_user_id', 1, inplace=True)

    # compare_data(test)

    # drop low feat_imps features
    if cfg.n_drop_feat_imps_cols and cfg.n_drop_feat_imps_cols > 0:
        feat_imps = pd.read_csv(cfg.feat_imps_path).sort_values(by='importance_mean', ascending=False).reset_index(drop=False)
        feat_imps_drops = feat_imps['feature'].iloc[-cfg.n_drop_feat_imps_cols:].values
        cols_drop_fi = [col for col in feat_cols if col in feat_imps_drops if col not in ['tweet_user_id']]
        train.drop(cols_drop_fi, axis=1, inplace=True)
        test.drop(cols_drop_fi, axis=1, inplace=True)
        print(f"Dropped {len(cols_drop_fi)} features on feature importance and add'l criteria")

    (feat_cols, media_img_feat_cols, text_feat_cols,
     user_des_feat_cols, user_img_feat_cols, feats_some) = get_feat_cols(train)
    # print(f"Some features list: {train[feats_some].columns}\n")



    cols2quantile_tfm = [col for col in train.columns if col in media_img_feat_cols+text_feat_cols
                                                    +user_img_feat_cols+user_des_feat_cols]
    if cfg.quantile_transform:
        train, test = transform_joint(train, test, cols2quantile_tfm,
                                      tfm=QuantileTransformer(n_quantiles=cfg.n_quantiles,
                                                              random_state=cfg.seed_other,
                                                              ))


    if cfg.impute_nulls:
        train = train.fillna(cfg.impute_value)
        test = test.fillna(cfg.impute_value)
        print(f"Imputed Nulls in train.py with {cfg.impute_value}")

    # standardize feats
    (feat_cols, media_img_feat_cols, text_feat_cols,
     user_des_feat_cols, user_img_feat_cols, feats_some) = get_feat_cols(train)

    categorical_columns_initial = [col for col in feat_cols if col.startswith('topic_id')
                                   # or col.startswith('tweets_in_')
                                    or col.startswith('tweet_language_id')
                                    or col.startswith('tweet_attachment_class')
                                    or col.startswith('ohe_')
                                   or col in ['user_has_location', 'tweet_has_attachment',
                                              'tweet_has_media', 'tweet_id_hthan1_binary','user_verified', 'user_has_url']
                                   ]

    print("Normalizing feats ....")
    if cfg.normalize_jointtraintest:
        cols2normalize = [col for col in feat_cols if
                          col not in categorical_columns_initial]  # get_cols2normalize(feat_cols)
        if cfg.quantile_transform:
            cols2normalize = [col for col in cols2normalize if col not in cols2quantile_tfm]
        is_nulls = (train[cols2normalize].isnull().sum().sum()>0).astype(bool)
        if is_nulls:
            train, test = normalize_npnan(train, test, cols2normalize)
        else:
            train, test = transform_joint(train, test, cols2normalize, tfm=StandardScaler())
        del cols2normalize

    if cfg.kbins_discretizer:
        cols2discretize_tfm = [col for col in train.columns if col in media_img_feat_cols + text_feat_cols
                               + user_img_feat_cols + user_des_feat_cols]
        train.loc[:,cols2discretize_tfm] = train.loc[:,cols2discretize_tfm].fillna(cfg.impute_value)
        test.loc[:, cols2discretize_tfm] = test.loc[:, cols2discretize_tfm].fillna(cfg.impute_value)
        train, test = transform_joint(train, test, cols2discretize_tfm,
                                      tfm=KBinsDiscretizer(n_bins=cfg.kbins_n_bins,
                                                           strategy=cfg.kbins_strategy, # {'uniform', 'quantile', 'kmeans'}
                                                           encode='ordinal'))
        print(f"KBinsDiscretize {len(cols2discretize_tfm)} cols, e.g. nunique 1 col of train: {train[cols2discretize_tfm[0]].nunique()}")

        del cols2discretize_tfm; _ = gc.collect()

    # cat columns for TABNET
    categorical_columns = []
    categorical_dims = {}
    len_train = len(train)
    train = pd.concat([train, test])
    for col in categorical_columns_initial:
        # print(col, train[col].nunique())
        l_enc = LabelEncoder()
        if cfg.model_arch_name=='tabnet':
            train[col] = train[col].fillna("VV_likely") # after normalize unlikely
        else:
            pass
            # train[col] = train[col].fillna(cfg.impute_value)
        train[col] = l_enc.fit_transform(train[col].values)
        categorical_columns.append(col)
        categorical_dims[col] = len(l_enc.classes_)
    test = train.iloc[len_train:]
    train = train.iloc[:len_train]
    cat_idxs = [i for i, f in enumerate(feat_cols) if f in categorical_columns]
    cat_dims = [categorical_dims[f] for i, f in enumerate(feat_cols) if f in categorical_columns]

    if cfg.extracted_feats_path and (cfg.extracted_feats_path.lower()!='none'):
        extracted_feats = pd.read_csv(cfg.extracted_feats_path)
        extracted_cols = [col for col in extracted_feats if col.startswith('extract_feature')]
        # assert train[['tweet_id','fold']].reset_index().equals(extracted_feats.loc[extracted_feats['is_test']==0, ['tweet_id','fold']].reset_index())
        train = train.merge(extracted_feats[['tweet_id']+extracted_cols], how='left', on='tweet_id')
        test = test.merge(extracted_feats[['tweet_id'] + extracted_cols], how='left', on='tweet_id')
        assert train[extracted_cols].isnull().sum().sum()==0, f"train[extracted_cols].isnull().sum().sum() = {train[extracted_cols].isnull().sum().sum()}"
        del extracted_feats; _ = gc.collect()
        print(f"Added {train[extracted_cols].shape[1]} extracted feature columns.")

    # print(f"train[categorical_columns].min():\n{train[categorical_columns].min()}")

    (feat_cols, media_img_feat_cols, text_feat_cols,
     user_des_feat_cols, user_img_feat_cols, feats_some) = get_feat_cols(train)

    # print("CHECK normality:")
    # print("train[feat_cols].mean().min() = {}, train[feat_cols].mean().max() = {}".format(
    #     train[feat_cols].mean().min(), train[feat_cols].mean().max()))
    # print("train[feat_cols].std().min() = {}, train[feat_cols].std().max() = {}".format(
    #     train[feat_cols].std().min(), train[feat_cols].std().max()))
    # print("test[feat_cols].mean().min() = {}, test[feat_cols].mean().max() = {}".format(
    #     test[feat_cols].mean().min(), test[feat_cols].mean().max()))
    # print("test[feat_cols].std().min() = {}, test[feat_cols].std().max() = {}".format(
    #     test[feat_cols].std().min(), test[feat_cols].std().max()))

    # print(f"Feature columns:\nmedia_img_feat_cols: n_cols = {len(media_img_feat_cols)}, {media_img_feat_cols[:1]}")
    # print(f"text_feat_cols: n_cols = {len(text_feat_cols)}, {text_feat_cols[:1]}")
    # print(f"user_des_feat_cols: n_cols = {len(user_des_feat_cols)}, {user_des_feat_cols[:1]}")
    # print(f"user_img_feat_cols: n_cols = {len(user_img_feat_cols)}, {user_img_feat_cols[:1]}")

    # print(f"Time: Normalized train and test in {gettime(t_normalize)}")
    logprint(f"Loaded and preprocessed train and test dataframesd in {gettime(t_get_data)}")

    train['virality'] -= 1

    print(f"Time to get data {gettime(t_get_data)}")
    preds_cols = [f"virality_{i}" for i in range(cfg.n_classes)]
    preds_val = pd.DataFrame(np.zeros((len(train), cfg.n_classes + 1)),
                             index=train.index)
    preds_val.columns = ['tweet_id'] + preds_cols
    preds_val['tweet_id'] = train['tweet_id'].values
    preds_test_soft = pd.DataFrame(np.zeros((len(test), cfg.n_classes + 1)), dtype=np.float32,
                                   columns=['tweet_id'] + preds_cols, index=test.index)
    preds_test_soft.loc[:, 'tweet_id'] = test['tweet_id'].values

    preds = pd.DataFrame(np.zeros((len(test), cfg.n_classes+1)))
    preds.columns = ['tweet_id'] + preds_cols
    preds.loc[:,'tweet_id'] = test['tweet_id'].values
    # preds_rank = pd.DataFrame(np.zeros((len(test), 1)))

    stats_all_ls, feat_imps_ls, preds_test_ls = [], [], []
    for fold in cfg.folds_to_train:
        t_start_fold = time.time()
        (preds_val_fold, preds_test_fold, stats_fold,
         feature_imp_fold_df) = run(fold, train, test, feat_cols,
                                 categorical_columns)
        preds_val.loc[train[train['fold']==fold].index, preds_cols] = preds_val_fold
        preds_test_soft.loc[:, preds_cols] += preds_test_fold / len(cfg.folds_to_train)
        preds_test_ls.append(pd.Series(np.argmax(preds_test_fold, axis=1)))
        stats_all_ls.append(stats_fold)
        logprint(f"Fold {fold} best acc train l score = {stats_fold['acc_train']:.5f}, with accuracy valid = {stats_fold['acc_valid']:.5f}")
        if fold==cfg.folds_to_train[0]:
            feat_imps_df = feature_imp_fold_df.copy()
        else:
            feat_imps_df.merge(feature_imp_fold_df, how='left', on='feature')
        print(f"Time Fold {fold} = {gettime(t_start_fold)}")


    print(f"===== Finished training, validating, predicting folds {cfg.folds_to_train} ======")
    stats_df = pd.DataFrame(stats_all_ls)
    stats_df.to_csv(osj(out_dir, 'stats.csv'), index=False)

    idxs = train[train['fold'].isin(cfg.folds_to_train)].index
    y_pred = np.argmax(preds_val.loc[idxs, preds_cols].values, axis=1)
    y_true = train.loc[idxs, 'virality'].values
    cv_score = accuracy_score(y_true, y_pred)
    preds_val.to_csv(osj(out_dir, 'preds_valid.csv'), index=False)
    average_score = stats_df['acc_valid'].mean()

    preds_test_cols = [f"preds_test_fold{i}" for i in cfg.folds_to_train]
    preds = pd.concat(preds_test_ls, axis=1)
    preds.index = test.index;
    preds.columns = preds_test_cols
    # preds = pd.concat([test[['tweet_id']], preds], axis=1)
    preds.to_csv(osj(out_dir, 'preds_test.csv'), index=False)
    preds_test_soft.to_csv(osj(out_dir, 'preds_test_soft.csv'), index=False)

    if cfg.save_submission:
        sub = pd.read_csv(osj(base_dir, 'solution_format.csv'), nrows=n_samples)
        # assert (sub['tweet_id'].values!=preds['tweet_id'].values).astype(int).sum()==0
        bin_count_preds = np.apply_along_axis(np.bincount, 1, preds[preds_test_cols].values, minlength=cfg.n_classes)
        sub['virality'] = np.argmax(bin_count_preds, axis=1)
        # sub['virality'] = np.argmax(preds[preds_test_cols].values, axis=1)
        sub['virality'] += 1
        # sub['virality'] = sub['virality'].astype(int)
        # print(f"submission.csv:\n{sub.head().to_string()}\n")
        # print(f"sub['virality'].value_counts():\n{sub['virality'].value_counts().to_string()}")
        sub.to_csv(osj(out_dir, f'submission_{int(round(cv_score*100_000,0))}.csv'), index=False)

        # sub['virality'] = preds_rank.values
        # sub['virality'] += 1
        # # sub['virality'] = sub['virality'].astype(int)
        # print(f"sub_rank['virality'].value_counts():\n{sub['virality'].value_counts().to_string()}")
        # sub.to_csv(osj(out_dir, f'submission_rank_{int(round(cv_score * 100_000, 0))}.csv'), index=False)


    print(f"Average score = {average_score*100:.4f}")
    print(f"CV score = {cv_score*100:.4f}")
    print(f"Total RUNTime = {gettime(t0)}")
    print(" ================= END OF CATBOOST BASE MODEL =================")

    # feat_imps = fnp.vstack(feat_imps_ls)
    # d_ = {'feature': feat_cols}
    # d_.update({f'fold_{i}': feat_imps[i,:] for i in range(feat_imps.shape[0])})
    # feat_imps_df = pd.DataFrame(d_)
    feat_imps_df['importance_mean'] = feat_imps_df.iloc[:,1:].mean(axis=1)
    # plot_mean_feature_importances(feat_imps_df[['feature', 'importance_mean']], dir2save=out_dir)
    feat_imps_df = feat_imps_df.sort_values(by='importance_mean', ascending=False)
    feat_imps_df.to_csv(osj(out_dir, 'feat_imps.csv'), index=False)

    # out_dir_new_name = rename_outdir_w_metric(out_dir, cv_score, ave_epoch=None)
    return

if __name__ == '__main__':

    t0 = time.time()
    base_dir = '../../data/raw'
    
    cfg = read_yaml() 
    cfg = addict.Dict(cfg)

    # overwrite cfg with args debug or seed
    args = parse_args()
    cfg.debug = False if args.debug.lower() == 'false' else True
    if cfg.seed_other != args.seed:
        cfg.seed_other = args.seed
    if cfg.seed_folds != args.seed:
        cfg.seed_folds = args.seed

    if cfg.debug:
        n_samples = cfg.n_samples_debug
    else:
        n_samples = int(1e12)

    out_dir, models_outdir = create_out_dir(cfg.experiment_name, cfg.model_arch_name,
                             cfg.n_folds, cfg.folds_to_train, cfg.debug)
    print(f"============\nStarting...\nout_dir: {out_dir}")

    copy_code(out_dir)
    # check model

    # seed_everything(cfg.seed_other)

    if cfg.debug:
        cfg.catboost_params.iterations=220
        cfg.folds_to_train = [0,1]

    # for key, val in cfg.items():
    #     print("{}: {}".format(key, val))

    main(out_dir, cfg)


