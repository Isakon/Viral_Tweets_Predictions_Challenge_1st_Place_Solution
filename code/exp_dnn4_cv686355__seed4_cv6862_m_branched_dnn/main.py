import os, glob, gc, time, yaml, shutil, random
import addict
import argparse

from collections import defaultdict
from tqdm import tqdm
import numpy as np 
import pandas as pd 

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelBinarizer, LabelEncoder, QuantileTransformer, KBinsDiscretizer
from datasets import (Features, transform_joint, normalize_npnan,
                      get_feat_cols, get_folds, save_preprocessed)
from datasets import ViralDataset
from utils import create_out_dir
from models import get_model_class


import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
# import apex
# from apex import amp
# import torchsummary

import warnings
warnings.filterwarnings('ignore')
pd.set_option("display.max_colwidth", 100)
pd.set_option("display.max_rows", 20)

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

def logprint_nulls(df, cols_group):
    if len(cols_group)>0:
        nongroup_cols = [col for col in df.columns if col not in cols_group]
        df_media_head = df[cols_group].isnull().sum().sort_values(ascending=False).head()
        df_nonmedia_head = df[nongroup_cols].isnull().sum().sort_values(ascending=False).head()
        print(
            f"Nulls in non_media columns df:\n{df_nonmedia_head}")
        print(
            f"Nulls in media columns df:\n{df_media_head}")

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # for faster training, but not deterministic

def get_class_weights(df, label='virality'):
    class_ratios = df[label].value_counts()/df.shape[0].sort_index()
    return class_ratios.to_list()

def get_criterion(loss_name):
    losses_zoo = {
        'bce': nn.CrossEntropyLoss(),
        'bce_weighted': nn.CrossEntropyLoss(weight=torch.tensor(cfg.dnn.loss_class_weights,
                                                                dtype=torch.float32)
                                                                .to(device)),
        'mse': nn.MSELoss()
    }  #  nn.BCEWithLogitsLoss()

    return losses_zoo[loss_name]

def accuracy_by_class(y_true, y_pred):
    acc_ls = []
    for y_t in np.sort(np.unique(y_true)):
        y_true_ova = np.where(y_true == y_t, 1, 0)
        y_pred_ova = np.where(y_pred == y_t, 1, 0)
        acc_y = accuracy_score(y_true_ova, y_pred_ova)
        acc_ls.append([y_t+1, acc_y])
    acc_arr = np.array(acc_ls)    
    acc_df = pd.DataFrame(acc_arr[:,1], index=acc_arr[:,0])
    # print(f"Accuracy by class:\n{acc_}")
    return acc_df

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

def save_soft_preds(preds: np.array, img_ids: list, fold=None):
    # prepare and save soft label predictions of test - submission_soft.csv
    preds_df = pd.DataFrame(preds, columns = [f"label{i}" for i in range(preds.shape[1])])
    test_img_ids_df = pd.DataFrame({"ImageID": img_ids})
    preds_df = pd.concat([test_img_ids_df, preds_df], axis=1).reset_index(drop=True)
    if fold:
        preds_df.to_csv(osj(out_dir, f'submission_soft_fold{fold}.csv'), index=False)
    else:
        preds_df.to_csv(osj(out_dir, f'submission_soft.csv'), index=False)

def save_submission_func(results_mean: np.array, test_img_ids: list):
    # prepare and save hard labels OFFICIAL submission
    submission = pd.DataFrame({"ImageID": test_img_ids, "label": [np.nan] * len(test_img_ids)})
    results_final_int = np.argmax(results_mean, axis=1)
    print(f"results_final_int.shape: {results_final_int.shape}\n\nresults_final[:3]:\n{results_mean[:2]}")

    class_to_label_mapping = {v: k for v, k in enumerate(dls.vocab)}
    print(f"{class_to_label_mapping}")
    results_final_str = [class_to_label_mapping[i] for i in results_final_int]
    submission['label'] = results_final_str
    # print(submission.shape)
    assert submission.isnull().sum().sum() == 0
    submission.to_csv(osj(out_dir, "submission.csv"), index=False)
    print("submission['label'].value_counts():\n{}".format(
                    submission['label'].value_counts().to_string()))


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
    if ave_epoch:
        ave_epoch = int(ave_epoch)
    prefix_ = f"cv{ave_metric_str}_"
    suffix_ = f"_e{ave_epoch}_cv{ave_metric_str}"
    new_base_name = prefix_ + os.path.basename(out_dir) + suffix_
    out_dir_new_name = osj(os.path.dirname(out_dir), new_base_name)
    # os.rename(out_dir, out_dir_new_name)
    assert not os.path.exists(out_dir_new_name), f"\nCan't rename: the path exists ({out_dir_new_name}"
    print(f"new out_dir directory name:\n{os.path.basename(out_dir_new_name)}")
    shutil.move(out_dir, out_dir_new_name)

    return out_dir_new_name

def train_epoch(model, optimizer, train_loader):
    model.train()
    bar = tqdm(train_loader)
    if cfg.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    losses = []
    # global_step = 0
    for batch_idx, (batch_data, targets) in enumerate(bar):

        batch_data, targets = batch_data.to(device), targets.to(device)  # , sample_weights.to(device)
        # import pdb; pdb.set_trace()
        if cfg.use_amp:
            with torch.cuda.amp.autocast():
                logits = model(batch_data)
                loss = criterion(logits, targets)
                # loss_arr = criterion(logits, targets)
                # loss = (loss_arr * sample_weights).mean()
                # if p['train']['grad_acc_steps'] > 1:
                #     loss = loss / p['train']['grad_acc_steps']
            scaler.scale(loss).backward()
            # if (batch_idx + 1) % p['train']['grad_acc_steps'] == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            # global_step += 1
        else:
            logits = model(batch_data)
            # import pdb;    pdb.set_trace()
            loss = criterion(logits, targets)
            # loss = (loss * sample_weights).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        losses.append(loss.item())

        # smooth_loss = np.mean(losses[-30:])

        # if is_interactive:
        #    bar.set_description(f'loss: {loss.item():.5f}')  # , smth: {smooth_loss:.5f}')

    loss_train = np.mean(losses)
    return loss_train


def valid_func(model, valid_loader):
    model.eval()
    bar = tqdm(valid_loader)

    PROB = []
    TARGETS = []
    losses = []
    PREDS = []

    with torch.no_grad():
        for batch_idx, (batch_data, targets) in enumerate(bar):
            batch_data, targets = batch_data.to(device), targets.to(device)
            logits = model(batch_data)
            PREDS += [logits.sigmoid()]
            TARGETS += [targets.detach().cpu()]
            loss = criterion(logits, targets)
            losses.append(loss.item())
            smooth_loss = np.mean(losses[-30:])
            bar.set_description(f'val loss: {loss.item():.5f}, val smooth loss: {smooth_loss:.5f}')

    PREDS = torch.cat(PREDS).cpu().numpy()
    TARGETS = torch.cat(TARGETS).cpu().numpy()
    # roc_auc = roc_auc_score(TARGETS.reshape(-1), PREDS.reshape(-1))
    acc = accuracy_score(TARGETS, np.argmax(PREDS, axis=1))
    loss_valid = np.mean(losses)
    return PREDS, loss_valid, acc   ## preds_val

def predict_test(model, test_loader):
    model.eval()
    bar = tqdm(test_loader)
    PREDS = []

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(bar):
            batch_data = batch_data.to(device)
            logits = model(batch_data)
            PREDS += [logits.sigmoid()]

    preds_test = torch.cat(PREDS).cpu().numpy()
    return preds_test


def run(fold, train, test, feats, feat_idxs):
    t_start_fold = time.time()
    target_col = 'virality'
    if cfg.debug:
        cfg.n_epochs = 3

    train_fold = train[train['fold'] != fold]  # .copy()
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
        drop_ids = adv_preds.loc[(adv_preds['is_test']==0) & (adv_preds['preds']<thresh), 'tweet_id'].values
        print(f"Before adversarial cutoff, train_fold.shape = {train_fold.shape}")
        train_fold = train_fold[~train_fold['tweet_id'].isin(drop_ids)]

    if cfg.mean_encoding:
        cols2encode = ['tweet_url_count']  # 'tweet_user_id
        cols2remove = []
        for col_ in cols2encode:
            if ('binned' not in col_) and (train_fold[col_].nunique()>100):
                train_fold[f"{col_}_binned"] = bin_feats(train_fold[col_],
                                                                feats=[col_],
                                                                n_bins_default=20)[f"{col_}_binned"]
                col2encode.remove(col_)
                cols2encode.append(f"{col_}_binned")
                cols2remove.append(col_)

        train_targets_ohe = LabelBinarizer().fit_transform(train_fold[target_col])
        train_targets_ohe = pd.DataFrame(train_targets_ohe, index = train_fold.index,
                                         columns = [f"target_ohe_{target_col}_{i}" for i in range(cfg.n_classes)]
                                         )
        for col2enc in cols2encode:
            for i, col_ in enumerate(train_targets_ohe.columns):
                target_means = train_targets_ohe.groupby(train_fold[col2enc])[col_].mean()
                new_col = f'meanenc_{col2enc}_{i}'
                train_fold[new_col] = train_fold[col2enc].map(target_means)
                val_fold[new_col] = val_fold[col2enc].map(target_means)
                test[new_col] = test[col2enc].map(target_means)
                train_fold[new_col] = train_fold[new_col].fillna(train_fold[new_col].mean())
                val_fold[new_col] = val_fold[new_col].fillna(val_fold[new_col].mean())
                test[new_col] = test[new_col].fillna(val_fold[new_col].mean())
                feats.extend([new_col])
        if len(cols2remove)>0:
            train_fold.drop(col2remove, axis=1, inplace=True)



    print(f"Train set n = {len(train_fold)}, Valid set n = {len(val_fold)}, Num feats = {train_fold.shape[1]}")
    dataset_train = ViralDataset(train_fold, feats, 'train')
    dataset_valid = ViralDataset(val_fold, feats, 'valid')
    dataset_test = ViralDataset(test, feats,  'test')
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=cfg.batch_size,
                                               shuffle=True, num_workers=cfg.num_workers,
                                               pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=cfg.batch_size*2,
                                               shuffle=False, num_workers=cfg.num_workers,
                                               )
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=cfg.batch_size * 2,
                                               shuffle=False, num_workers=cfg.num_workers,
                                               )
    print(f"{time.ctime()} ===== Fold {fold}/{cfg.n_folds-1} starting ======")

    # if cfg.model_arch_name in ['simple_dnn']:
    model = ModelClass(input_dim=train_fold[feats].shape[1],
                       output_dim=cfg.n_classes,
                       layers_dims=cfg.dnn.layers,
                       dropouts=cfg.dnn.dropouts,
                       feat_idxs=feat_idxs)
    assert len(feats)==np.sum([len(c) for k,c in feat_idxs.items()])
    # torchsummary.summary(model, (265, len(feats)))
    if fold==cfg.folds_to_train[0]: print(model)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.init_lr,
                                 weight_decay=cfg.weight_decay)
    # if cfg.fp16:
    #     model, optimizer = amp.initialize()

    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.n_epochs, eta_min=1e-8)

    model_file = os.path.join(osj(models_outdir, f'model_best_fold{fold}.pth'))
    stats = defaultdict(list)
    not_improving = 0
    best_acc = 0
    for epoch in range(cfg.n_epochs):
        t_epoch = time.time()
        content = f"\nFold {fold}:\tEpoch {epoch}/{cfg.n_epochs-1}:"
        print(content)
        loss_train = train_epoch(model, optimizer, train_loader)
        scheduler.step()
        preds_val_epoch, loss_valid, acc_epoch = valid_func(model, valid_loader)
        if acc_epoch > best_acc:
            print(f'best accuracy ({best_acc:.5f} --> {acc_epoch:.5f}). Saving model ...')
            if cfg.save_preds_model:
                save_model(model_file,  model, epoch, acc_epoch, save_weights_only=True)
            best_acc = acc_epoch
            preds_val_fold = preds_val_epoch
            not_improving = 0

        lr = optimizer.param_groups[0]['lr']
        stats['epoch'].append(epoch)
        stats['lr'].append(lr)
        stats['loss_train'].append(loss_train)
        stats['loss_valid'].append(loss_valid)
        stats['score'].append(acc_epoch*100)

        #content = f'lr_base: {stats["lr_base"][-1]:.7f}, loss_train: {loss_train:.5f}, loss_valid: {loss_valid:.5f}'
        content = f'\nlr: {lr:7f}, loss_train: {loss_train:.5f}, loss_valid: {loss_valid:.5f}, accuracy: {acc_epoch:.5f}'
        print(content)

        print(f"ACCURACY: {acc_epoch: .5f} \tepoch_duration = {gettime(t_epoch)}\n\n {'-' * 30}")

        if not_improving == cfg.early_stop:
            print('Early Stopping...')
            break

    stats_fold_df = pd.DataFrame(stats)
    stats_fold_df['fold'] = fold
    # stats_fold_df.to_csv(osj(out_dir, f'stats_fold_{fold}.csv'), index=False)

    if cfg.predict_test and cfg.save_preds_model:
        checkpoint = torch.load(model_file, map_location=device)
        model.load_state_dict(checkpoint['model'])
        del checkpoint; _ = gc.collect()
        print(f"Predicting test set ...")
        preds_test_fold = predict_test(model, test_loader)
    else:
        preds_test_fold = None

    return preds_val_fold, preds_test_fold, stats_fold_df

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
    print(f"Dropped {n - train.shape[0]} duplicated rows from train. train.shape[0] = {train.shape}")
    return train

def main(out_dir, cfg):
    t_get_data = time.time()
    # assert os.path.isfile(cfg.path_train_duplicates)
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

        # saving and loading preprocessed data in order to reproduce the score
        # if cfg.save_train_test:  # and (not cfg.debug):  # "../../data/preprocessed/"
        save_preprocessed(cfg, train, test, path_train=cfg.train_preprocessed_path,
                          path_test=cfg.test_preprocessed_path)
        train = pd.read_csv(cfg.train_preprocessed_path, nrows=n_samples)
        test = pd.read_csv(cfg.test_preprocessed_path, nrows=n_samples)
        # os.remove(cfg.train_preprocessed_path)
        # os.remove(cfg.train_preprocessed_path)

        # if cfg.save_train_test:  #  and (not cfg.debug):  # "../../data/preprocessed/"
        #     save_preprocessed(cfg, train, test, path_train=cfg.train_preprocessed_path,
        #                       path_test=cfg.test_preprocessed_path)


    train = get_folds(cfg, train)
    (feat_cols, media_img_feat_cols, text_feat_cols,
    user_des_feat_cols, user_img_feat_cols, feats_some) = get_feat_cols(train)

    # compare_data(test)

    if cfg.drop_tweet_user_id:
        if 'tweet_user_id' in train.columns:
            train.drop('tweet_user_id', 1, inplace=True)
            test.drop('tweet_user_id', 1, inplace=True)

    # drop low feat_imps features
    if cfg.n_drop_feat_imps_cols and cfg.n_drop_feat_imps_cols > 0:
        feat_imps = pd.read_csv(cfg.feat_imps_path).sort_values(by='importance_mean', ascending=False).reset_index(drop=False)
        feat_imps_drops = feat_imps['feature'].iloc[-cfg.n_drop_feat_imps_cols:].values
        cols_drop_fi =  [col for col in feat_cols if col in feat_imps_drops if col not in ['tweet_user_id']]
        train.drop(cols_drop_fi, axis=1, inplace=True)
        test.drop(cols_drop_fi, axis=1, inplace=True)
        print(f"Dropped {len(cols_drop_fi)} features on feature importance")
    (feat_cols, media_img_feat_cols, text_feat_cols,
     user_des_feat_cols, user_img_feat_cols, feats_some) = get_feat_cols(train)
    # print(f"Some features list: {train[feats_some].columns}\n")

    cols2quantile_tfm = [col for col in train.columns if col in media_img_feat_cols + text_feat_cols
                         + user_img_feat_cols + user_des_feat_cols]
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
                                              'tweet_has_media', 'tweet_id_hthan1_binary', 'user_verified',
                                              'user_has_url']
                                   ]

    print("Normalizing feats ....")
    if cfg.normalize_jointtraintest:
        cols2normalize = [col for col in feat_cols if
                          col not in categorical_columns_initial]  # get_cols2normalize(feat_cols)
        if cfg.quantile_transform:
            cols2normalize = [col for col in cols2normalize if col not in cols2quantile_tfm]
        is_nulls = (train[cols2normalize].isnull().sum().sum() > 0).astype(bool)
        if is_nulls:
            train, test = normalize_npnan(train, test, cols2normalize)
        else:
            train, test = transform_joint(train, test, cols2normalize, tfm=StandardScaler())
        del cols2normalize

    if cfg.kbins_discretizer:
        cols2discretize_tfm = [col for col in train.columns if col in media_img_feat_cols + text_feat_cols
                               + user_img_feat_cols + user_des_feat_cols]
        train.loc[:, cols2discretize_tfm] = train.loc[:, cols2discretize_tfm].fillna(cfg.impute_value)
        test.loc[:, cols2discretize_tfm] = test.loc[:, cols2discretize_tfm].fillna(cfg.impute_value)
        train, test = transform_joint(train, test, cols2discretize_tfm,
                                      tfm=KBinsDiscretizer(n_bins=cfg.kbins_n_bins,
                                                           strategy=cfg.kbins_strategy,
                                                           # {'uniform', 'quantile', 'kmeans'}
                                                           encode='ordinal'))
        print(
            f"KBinsDiscretize {len(cols2discretize_tfm)} cols, e.g. nunique 1 col of train: {train[cols2discretize_tfm[0]].nunique()}")

        del cols2discretize_tfm;
        _ = gc.collect()
    if cfg.extracted_feats_path and (cfg.extracted_feats_path.lower()!='none'):
        extracted_feats = pd.read_csv(cfg.extracted_feats_path)
        extracted_cols = [col for col in extracted_feats if col.startswith('extract_feature')]
        # assert train[['tweet_id','fold']].reset_index().equals(extracted_feats.loc[extracted_feats['is_test']==0, ['tweet_id','fold']].reset_index())
        train = train.merge(extracted_feats[['tweet_id']+extracted_cols], how='left', on='tweet_id')
        test = test.merge(extracted_feats[['tweet_id'] + extracted_cols], how='left', on='tweet_id')
        assert train[extracted_cols].isnull().sum().sum()==0, f"train[extracted_cols].isnull().sum().sum() = {train[extracted_cols].isnull().sum().sum()}"
        del extracted_feats; _ = gc.collect()
        print(f"Added {train[extracted_cols].shape[1]} extracted feature columns.")

    (feat_cols, media_img_feat_cols, text_feat_cols,
     user_des_feat_cols, user_img_feat_cols, feats_some) = get_feat_cols(train)
    # cat columns for TABNET, catboost, lightgbm
    categorical_columns = []
    categorical_dims = {}
    len_train = len(train)
    train = pd.concat([train, test])
    for col in categorical_columns_initial:
        # print(col, train[col].nunique())
        l_enc = LabelEncoder()
        if cfg.model_arch_name == 'tabnet':
            train[col] = train[col].fillna("VV_likely")  # after normalize unlikely
        else:
            pass
            # train[col] = train[col].fillna(cfg.impute_value)
        train[col] = l_enc.fit_transform(train[col].values)
        categorical_columns.append(col)
        categorical_dims[col] = len(l_enc.classes_)
    test = train.iloc[len_train:]
    train = train.iloc[:len_train]
    cat_idxs = [i for i, f in enumerate(feat_cols) if f in categorical_columns]
    # cat_dims = [categorical_dims[f] for i, f in enumerate(feat_cols) if f in categorical_columns]
    feat_idxs = {}
    feat_idxs['category_feats'] = cat_idxs
    feat_idxs['numerical_feats'] = [i for i,f in enumerate(feat_cols)
                    if f not in media_img_feat_cols+text_feat_cols+user_img_feat_cols+user_des_feat_cols
                                    +user_img_feat_cols+categorical_columns]
    feat_idxs['text_idxs'] = [i for i,f in enumerate(feat_cols) if f in text_feat_cols]
    feat_idxs['media_img_feats'] = [i for i, f in enumerate(feat_cols) if f in media_img_feat_cols]
    feat_idxs['user_img_feats'] = [i for i, f in enumerate(feat_cols) if f in user_img_feat_cols]
    feat_idxs['user_des_feats'] = [i for i, f in enumerate(feat_cols) if f in user_des_feat_cols]



    print("CHECK normality:")
    print("train[feat_cols].mean().min() = {}, train[feat_cols].mean().max() = {}".format(
        train[feat_cols].mean().min(), train[feat_cols].mean().max()))
    print("train[feat_cols].std().min() = {}, train[feat_cols].std().max() = {}".format(
        train[feat_cols].std().min(), train[feat_cols].std().max()))

    # print(f"Feature columns:\nmedia_img_feat_cols: n_cols = {len(media_img_feat_cols)}, {media_img_feat_cols[:1]}")
    # print(f"text_feat_cols: n_cols = {len(text_feat_cols)}, {text_feat_cols[:1]}")
    # print(f"user_des_feat_cols: n_cols = {len(user_des_feat_cols)}, {user_des_feat_cols[:1]}")
    # print(f"user_img_feat_cols: n_cols = {len(user_img_feat_cols)}, {user_img_feat_cols[:1]}")

    # end of loading and preprocessing

    train['virality'] -= 1

    print(f"Time to get data {gettime(t_get_data)}")
    preds_cols = [f"virality_{i}" for i in range(cfg.n_classes)]
    preds_val = pd.DataFrame(np.zeros((len(train), cfg.n_classes+1)),
                             index=train.index)
    preds_val.columns = ['tweet_id'] + preds_cols
    preds_val['tweet_id'] = train['tweet_id'].values

    preds_test_soft = pd.DataFrame(np.zeros((len(test), cfg.n_classes + 1)), dtype=np.float32,
                                   columns=['tweet_id'] + preds_cols, index=test.index)
    preds_test_soft.loc[:, 'tweet_id'] = test['tweet_id'].values

    best_fold_stats_ls, stats_all_ls, preds_test_ls = [], [], []
    for fold in cfg.folds_to_train:
        t_start_fold = time.time()
        preds_val_fold, preds_test_fold, stats_fold_df = run(fold, train, test, feat_cols, feat_idxs)
        preds_val.loc[train[train['fold']==fold].index, preds_cols] = preds_val_fold
        preds_test_soft.loc[:, preds_cols] += preds_test_fold / len(cfg.folds_to_train)
        preds_test_ls.append(pd.Series(np.argmax(preds_test_fold, axis=1)))
        stats_all_ls.append(stats_fold_df)
        best_fold_stats = stats_fold_df[stats_fold_df['score']==stats_fold_df['score'].max()]
        best_fold_stats_ls.append(best_fold_stats)
        logprint(f"Fold {fold} best accuracy = {best_fold_stats['score'].iloc[0]:.5f}, at epoch {best_fold_stats['epoch'].iloc[0]} with valid loss = {best_fold_stats['loss_valid'].iloc[0]:.5f}")
        # if cfg.predict_test:
        #     preds.loc[:, preds_cols] += preds_test_fold / len(cfg.folds_to_train)
        print(f"Time Fold {fold} = {gettime(t_start_fold)}")


    print(f"===== Finished training, validating, predicting folds {cfg.folds_to_train} ======")
    stats_df = pd.concat(stats_all_ls)
    stats_df.to_csv(osj(out_dir, 'stats.csv'), index=False)

    summary_df = pd.concat(best_fold_stats_ls)
    summary_df.to_csv(osj(out_dir, 'a_summary.csv'), index=False)

    preds_val.to_csv(osj(out_dir, 'preds_valid.csv'), index=False)
    idxs = train[train['fold'].isin(cfg.folds_to_train)].index
    y_pred = np.argmax(preds_val.loc[idxs, preds_cols].values, axis=1)
    y_true = train.loc[idxs, 'virality'].values
    cv_score = accuracy_score(y_true, y_pred)
    acc_df = accuracy_by_class(y_true, y_pred)
    #print(f"class_scores:\n{class_scores_df.to_string()}")
    average_score = summary_df['score'].mean()
    if average_score>68.20 or cfg.debug:  # cfg.save_preds and cfg.predict_test:
        preds_test_cols = [f"preds_test_fold{i}" for i in cfg.folds_to_train]
        preds = pd.concat(preds_test_ls, axis=1)
        preds.index = test.index;
        preds.columns = preds_test_cols
        # preds = pd.concat([test[['tweet_id']], preds], axis=1)
        preds.to_csv(osj(out_dir, 'preds_test.csv'), index=False)
        preds_test_soft.to_csv(osj(out_dir, 'preds_test_soft.csv'), index=False)

        sub = pd.read_csv(osj(base_dir, 'solution_format.csv'), nrows=n_samples)
        # assert (sub['tweet_id'].values!=preds['tweet_id'].values).astype(int).sum()==0
        bin_count_preds = np.apply_along_axis(np.bincount, 1, preds[preds_test_cols].values, minlength=cfg.n_classes)
        sub['virality'] = np.argmax(bin_count_preds, axis=1)
        sub['virality'] += 1
        # sub['virality'] = sub['virality'].astype(int)
        #print(f"submission.csv:\n{sub.head().to_string()}\n")
        train_vc = ((train['virality']+1).value_counts()/train.shape[0]).round(4)
        sub_vc = (sub['virality'].value_counts()/sub.shape[0]).round(4)
        vc_ = pd.concat([sub_vc, train_vc, acc_df], axis=1)
        vc_.columns = ['sub', 'train', 'accuracy']
        print(f"sub and train ['virality'].value_counts():\n{vc_.to_string()}")
        sub.to_csv(osj(out_dir, f'submission_{int(round(cv_score*100_000,0))}.csv'), index=False)

    print(f"Average score = {average_score:.4f}")
    print(f"CV score = {cv_score*100:.4f}")
    print(f"Total RUNTime = {gettime(t0)}")
    print(" ================= END OF DNN4 seed=4 BASE MODEL =================")

    # _ = rename_outdir_w_metric(out_dir, cv_score, ave_epoch=summary_df['epoch'].mean())
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

    remove_from_cfg = ('dnn_extract_feats', 'lgb_model_params', 'tabnet', 'lgb_fit_params',
                       'tabnet_fit', 'xgb_model_params', 'xgb_fit_params', 'catboost_params')
    for k in remove_from_cfg:
        cfg.pop(k, None)

    out_dir, models_outdir = create_out_dir(cfg)
    print(f"============\nStarting...\nout_dir: {out_dir}")

    copy_code(out_dir)
    # check model
    ModelClass = get_model_class(cfg.model_arch_name)
    device = torch.device('cuda:0')

    criterion = get_criterion(cfg.dnn.loss_name)
    seed_everything(cfg.seed_other)

    # model_check = ModelClass(input_dim=100, output_dim=8,
    #                          layers_dims=cfg.dnn.layers, dropouts=cfg.dnn.dropouts,
    #                          feat_idxs={feat_idxs)
    # print(f"Model:\n{model_check}")
    # del model_check; _ = gc.collect()

    cfg.impute_nulls = True # for NNs
    # cfg.impute_value should be given

    if cfg.debug:
        cfg.n_epochs=2
        cfg.folds_to_train = [0,1]

    # for key, val in cfg.items():
    #     print("{}: {}".format(key, val))

    main(out_dir, cfg)


