import os, time
import numpy as np
import pandas as pd

osj = os.path.join


def create_out_dir(cfg, prefix=''):
    # datetime_str = time.strftime("%d_%m_time_%H_%M", time.localtime())
    # folds_str = '_'.join([str(fold) for fold in cfg.folds_to_train])
    # out_dir = '../../submissions/{}_{}_m_{}_ep{}_bs{}_nf{}_t_{}'.format(
    #             prefix, cfg.experiment_name, cfg.model_arch_name, cfg.n_epochs,
    #     cfg.batch_size, cfg.n_folds,  datetime_str)   # bs, weight_decay, , folds_str,
    #
    # if cfg.debug:
    #     out_dir = osj(os.path.dirname(out_dir), 'debug_' + os.path.basename(out_dir))
    out_dir = cfg.out_dir
    models_outdir = osj(out_dir, 'models')
    os.makedirs(out_dir)
    os.makedirs(models_outdir)
    return out_dir, models_outdir

def add_pseudo_virality_col(test, pseudo_path_, n_samples, from_fold_preds=False):
    if from_fold_preds:
        preds_test = np.load(pseudo_path_)[:n_samples]
        test['virality'] = np.argmax(preds_test, axis=1)
    else:
        sub = pd.read_csv(pseudo_path_, nrows=n_samples)
        test['virality'] = sub['virality']
        assert test['tweet_id'].reset_index(drop=True).equals(sub['tweet_id'].reset_index(drop=True))
    return test
