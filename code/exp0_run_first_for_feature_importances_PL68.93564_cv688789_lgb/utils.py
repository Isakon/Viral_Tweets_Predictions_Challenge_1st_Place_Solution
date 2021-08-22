import os
import numpy as np
import pandas as pd

def compare_data(test,
                 compare_path='/home/isakev/challenges/viral_tweets/data/processed/test_date_tweet2user_ratios_quarterly_dnn_jun22.csv'):
    print("preprocessed path:\n", compare_path)
    # print("cfg.test_preprocessed_path\n:", cfg.test_preprocessed_path)
    # assert os.path.basename(compare_path) == os.path.basename(cfg.test_preprocessed_path)

    test_compare = pd.read_csv(compare_path)

    cols_here_not_in_preproc = set(test.columns).difference(set(test_compare.columns))
    cols_preproc_not_in_here = set(test_compare.columns).difference(set(test.columns))

    print("cols_preproc_not_in_here:\n", cols_preproc_not_in_here)
    print()
    print("cols_here_not_in_preproc:\n", cols_here_not_in_preproc)

    print("test.isnull().sum().sort_values().tail():\n", test.isnull().sum().sort_values().tail())
    print("\ntest_compare.isnull().sum().sort_values().tail():\n", test_compare.isnull().sum().sort_values().tail())
    print()
    minus_ones_compare, minus_ones_here = [], []
    for col in test_compare.columns:
        minus_ones_compare.append((test_compare[col] == -1).sum())
        minus_ones_here.append((test_compare[col] == -1).sum())
    print("minus_ones_compare:", sum(minus_ones_compare))
    print("minus_ones_here:", sum(minus_ones_here))
    print()

    assert len(cols_preproc_not_in_here) == 0
    assert len(cols_preproc_not_in_here) == 0
    if len(test)>5000:
        assert len(test) == len(test_compare), f"len test = {len(test)} , len test_compare = {len(test_compare)}"
    assert sum(minus_ones_compare) == sum(minus_ones_here)
    min_len = min(len(test), len(test_compare))
    test = test.iloc[:min_len].reset_index(drop=True)
    test_compare = test_compare[:min_len]
    unequals = test.compare(test_compare)
    print("test.compare(test_compare).shape[1]//2", unequals.shape[1]//2, "test.shape", test.shape)
    print("test.compare(test_compare).columns", unequals.columns)

    diffs_ls = []
    for col0 in unequals.columns.get_level_values(0):
        diffs = np.isclose(unequals[(col0, 'self')], unequals[(col0, 'other')])
        diffs_ls.append(np.sum(diffs)/len(diffs))

    argsorted_cols = unequals.columns.get_level_values(0)[np.argsort(diffs_ls)]

    print("np.sum(diffs_ls", np.sum(diffs_ls))
    print("some  diffs_ls[-10:]",
          [f"{col}: {diff_}" for (col, diff_) in
                                zip(argsorted_cols[-10:], np.sort(diffs_ls)[-10:])])

    # assert test.compare(test_compare).shape[1] == 0, "test.compare(test_compare).shape[1] == 0"