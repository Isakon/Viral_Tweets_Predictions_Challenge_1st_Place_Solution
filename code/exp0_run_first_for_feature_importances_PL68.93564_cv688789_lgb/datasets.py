import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from joblib import parallel_backend
from multiprocessing import cpu_count
import os, gc, joblib
from tqdm import tqdm
from collections import defaultdict
import torch

import warnings
warnings.filterwarnings('ignore')
pd.set_option("display.max_colwidth", 100)
pd.set_option("display.max_rows", 20)

osj = os.path.join; osl = os.listdir
n_cpus = cpu_count()

class ViralDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, feat_cols: list, mode: str):
        self.X = df[feat_cols].values  # [:,np.newaxis,:]
        self.mode = mode

        if mode != 'test':
            self.targets = df['virality'].values  # [:,np.newaxis]  # - 1
            # assert np.sum(~df['virality'].isin(list(range(5))))==0

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.mode=='test':
            return torch.tensor(self.X[idx], dtype=torch.float32)
        else:
            return (torch.tensor(self.X[idx], dtype=torch.float32),
                    torch.tensor(self.targets[idx], dtype=torch.long))  # long))

class ExtractFeatsDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, feat_cols: list, target_cols: list, mode: str):
        self.X = df[feat_cols].values  # [:,np.newaxis,:]
        # self.target_cols = target_cols
        self.mode = mode

        if mode != 'test':
            if len(target_cols)==1:
                self.targets = df[target_cols[0]].values  # [:,np.newaxis]  # - 1
                self.target_dtype = torch.long
            else:
                self.targets = df[target_cols].values  # [:,np.newaxis]  # - 1
                self.target_dtype = torch.float32
            # assert np.sum(~df['virality'].isin(list(range(5))))==0

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.mode=='test':
            return torch.tensor(self.X[idx], dtype=torch.float32)
        else:
            return (torch.tensor(self.X[idx], dtype=torch.float32),
                    torch.tensor(self.targets[idx], dtype=self.target_dtype))  # long))


def to_binary_categories(df, cat_col='tweet_language_id'):
    df.loc[:, cat_col] = (df[cat_col]!=0).astype(np.int8)
    return df

def freq_encoding(df, freq_cols: list, main_col='tweet_id'):
    for c in freq_cols:
        count_df = df.groupby([c])[main_col].count().reset_index()
        count_df.columns = [c, '{}_freq'.format(c)]
        df = df.merge(count_df, how='left', on=c)
    return df


def bin_feats(df, feats=[], n_bins_default=20):
    bin_counts = defaultdict(lambda: n_bins_default)
    bin_counts['user_tweet_count'] = 20
    for feature in feats:
        if '_binned' in feature:
            continue
        n_bins = bin_counts[feature]
        if n_bins:
            bins = np.unique(df[feature].quantile(np.linspace(0, 1, n_bins)).values)
            df[feature + '_binned'] = pd.cut(
                df[feature], bins=bins, duplicates='drop'
            ).cat.codes
    return df

def to_categorical(df):
    cat_cols = ['tweet_has_attachment', 'user_has_location', 'user_has_url', 'user_verified', ]
    df[cat_cols] = df[cat_cols].astype('category')
    return df

def change2float32(df):
    float_cols = df.select_dtypes('float64').columns
    df[float_cols] = df[float_cols].astype(np.float32)
    return df

def merge_df2media(df, df_media):
    num_media = (df_media.groupby('tweet_id')['media_id']
                 .nunique()
                 .reset_index())
    df_media.drop('media_id', axis=1, inplace=True)
    num_media.columns = ['tweet_id', 'num_media']
    df_media = df_media.merge(num_media, how='left', on='tweet_id')
    media_cols = [col for col in df_media if col not in ['tweet_id','media_id']]
    df_media = df_media.groupby('tweet_id')[media_cols].mean().reset_index()
    # df_media = mean_feats.merge(df_media[['tweet_id']], how='left', on='tweet_id')
    # del mean_feats; _ = gc.collect()
    df_media['tweet_has_media'] = True
    df = df.merge(df_media, how='left', on='tweet_id')
    # fillna False if tweet has no media
    df['tweet_has_media'] = df['tweet_has_media'].fillna(False)
    # the same for the count of number of media per tweet
    df['num_media'] = df['num_media'].fillna(0).astype(np.int8)
    return df

def tweets_user_created_date(df):
    for feat_ in ['tweet_created_at_year', 'tweet_created_at_month', 'tweet_created_at_day',
                  'tweet_created_at_hour']:
        # counts_df_cols = ['tweet_user_id']+[f"tweets_in_{feat_.split('_')[-1]}_{time_}" for time_ in np.sort(df[feat_].unique())]
        # tweet_user_ids = np.sort(df['tweet_user_id'].unique())
        # counts_df = pd.DataFrame(index=range(tweet_user_ids), columns=counts_df_cols)
        # counts_df['tweet_user_id'] = tweet_user_ids
        counts_map = df.groupby('tweet_user_id')[feat_].apply(lambda x: x.value_counts())
        counts_map = counts_map.unstack(level=1)
        counts_map.columns = [f"tweets_in_{feat_.split('_')[-1]}_"+str(col) for col in counts_map.columns]
        counts_map = counts_map.fillna(0).reset_index()
        df = df.merge(counts_map, how='left', on='tweet_user_id')
    return df

def create_date_col(df):
    tweet_date_cols = ['tweet_created_at_year', 'tweet_created_at_month', 'tweet_created_at_day']
    df['date'] = df[tweet_date_cols].apply(lambda x:
                                           str(x['tweet_created_at_month']).strip() + '/' +
                                           str(x['tweet_created_at_day']).strip() + '/' +
                                           str(x['tweet_created_at_year']).strip(), axis=1)
    df['date'] = pd.to_datetime(df['date'])
    return df

def add_date_feats(df, impute_nulls=False, impute_value=0):
    # todo reinstate index - recover df sorted by date
    #df_old_index = df.index
    df = create_date_col(df)

    cols_resample = ['tweet_hashtag_count',  'tweet_url_count',  'tweet_mention_count',
                     ]
    date_freqs = ['1Q']  # ,'1M']
    stats = ['sum','mean','std','min','max']  # ['mean', 'max', 'min', 'median', 'std']
    for freq_ in date_freqs:
        for stat_ in stats:
            df.set_index('date', inplace=True)
            g = (df.groupby('tweet_user_id').resample(freq_, closed='left')
                    [cols_resample].agg(stat_)
                            .astype(np.float32)
                 )  #                .set_index('date'))
            g = g.unstack('date').fillna(0)
            g.columns = [col1 + f'_func_{stat_}_' + col2.strftime('%Y-%m-%d') for (col1, col2) in g.columns]
            g.reset_index(inplace=True)
            # g = g.rename(columns ={col: f"{col}_rsmpl_{freq_}_func_{stat_}"
            #                                for col in g.columns if col not in ['tweet_user_id','date']})
            #df = df.reset_index().merge(g, how='left', on='tweet_user_id')
            df = df.reset_index().merge(g, how='left', on='tweet_user_id')

    # df.reset_index(drop=False, inplace=True)

# todo  count 'tweet_id' for each period for user

    today = pd.to_datetime('7/1/2021')

    df['days_since_tweet'] = (today - df['date']).dt.days  # .astype(int)
    df['user_followers_count_2days'] = df['user_followers_count'] / df['days_since_tweet']
    df['user_following_count_2days'] = df['user_following_count'] / df['days_since_tweet']
    df['user_listed_on_count_2days'] = df['user_listed_on_count'] / df['days_since_tweet']
    df['user_tweet_count_2days'] = df['user_tweet_count'] / df['days_since_tweet']
    df['tweet_hashtag_count_2days'] = df['tweet_hashtag_count'] / df['days_since_tweet']
    df['tweet_mention_count_2days'] = df['tweet_mention_count'] / df['days_since_tweet']
    df['tweet_url_count_2days'] = df['tweet_url_count'] / df['days_since_tweet']

    # todo not a date related functions:
    df['tweet_mention_count_div_followers'] = df['tweet_mention_count'].divide(df['user_followers_count']+1)
    df['tweet_url_count_div_followers'] = df['tweet_url_count'].divide(df['user_followers_count']+1)
    df['tweet_hashtag_count_div_followers'] = df['tweet_hashtag_count'].divide(df['user_followers_count']+1)
    df['tweet_mention_count_div_followers'] = df['tweet_mention_count'].divide(df['user_followers_count']+1)

    df['tweet_mention_count_div_n_tweets'] = df['tweet_mention_count'].divide(df['user_tweet_count']+1)
    df['tweet_url_count_div_n_tweets'] = df['tweet_url_count'].divide(df['user_tweet_count']+1)
    df['tweet_hashtag_count_div_n_tweets'] = df['tweet_hashtag_count'].divide(df['user_tweet_count']+1)
    df['tweet_mention_count_div_n_tweets'] = df['tweet_mention_count'].divide(df['user_tweet_count']+1)

    df['tweet_mention_count_div_likes'] = df['tweet_mention_count'].divide(df['user_like_count']+1)
    df['tweet_url_count_div_likes'] = df['tweet_url_count'].divide(df['user_like_count']+1)
    df['tweet_hashtag_count_div_likes'] = df['tweet_hashtag_count'].divide(df['user_like_count']+1)
    df['tweet_mention_count_div_likes'] = df['tweet_mention_count'].divide(df['user_like_count']+1)

    cols_drop = ['date', 'tweet_created_at_year', 'tweet_created_at_month',
                 'tweet_created_at_day',
                 'user_created_at_year', 'user_created_at_month']
    df.drop(cols_drop, axis=1, inplace=True)

    return df


def ohe_func(df, cat_col, ohe_tfm=LabelBinarizer(), prefix=None):
    """ OHE one categorical column of df, and return df with columns 'label_{range(1,x}' added
    """
    # ohe.iloc[:, df['tweet_language_id'].tolist()]
    ohe_tfm.fit(df[cat_col])
    ohe_transformed = ohe_tfm.transform(df[cat_col])
    if prefix:
        cat_cols = [f'{prefix}_{cat_col}_{i}' for i in range(ohe_transformed.shape[1])]
    else:
        cat_cols = [f'{cat_col}_{i}' for i in range(ohe_transformed.shape[1])]

    ohe_df = pd.DataFrame(ohe_transformed, index=df.index, columns=cat_cols)
    df = pd.concat([df, ohe_df], axis=1)
    df.drop(cat_col, axis=1, inplace=True)
    return df


class Features():
    def __init__(self,):
        self.transformers = {}
        self.impute_img_feature_nulls = -1
        self.media_img_feat_cols = []
        self.text_feat_cols =  []
        self.user_des_feat_cols =  []
        self.user_img_feat_cols =  []
        # union of topic ids in train and test , 0 - nan value, min=36, max=172
        # xor train, test = [ 38, 117, 123, 165]
        # in test but not in train = [ 38, 117, 123]
        self.unique_topic_ids = [  0,  36,  37,  38,  39,  43,  44,  45,  52,  58,  59,  60,  61,
         63,  68,  71,  72,  73,  78,  79,  80,  81,  82,  87,  88,  89,
         91,  93,  98,  99, 100, 101, 104, 111, 112, 117, 118, 119, 120,
        121, 122, 123, 125, 126, 127, 147, 148, 149, 150, 151, 152, 153,
        155, 156, 163, 165, 169, 170, 171, 172]

        self.cols2int8 = ['fold', 'user_created_at_month', 'tweet_created_at_day', 'tweet_created_at_hour',
         'tweet_hashtag_count', 'tweet_url_count',  'tweet_mention_count', 'tweet_has_attachment',
         'virality', 'tweet_has_media', 'user_has_url', 'user_verified', 'num_media',
                          'user_id', 'tweet_user_id']
        # 'tweet_created_at_year', 'user_created_at_year',
        self.cols2int8 += [f'tweet_language_id_{i}' for i in range(30)]

    def get_data_stage1(self, cfg, base_dir, n_samples=int(1e10)):
        df = pd.read_csv(osj(base_dir, 'Tweets',f'train_tweets.csv'), nrows=n_samples)
        test = pd.read_csv(osj(base_dir, 'Tweets',f'test_tweets.csv'), nrows=n_samples)
        test_tweet_ids = test['tweet_id'].to_list()
        # self.tabular_feats.append()
        df = pd.concat([df, test])
        del test; _ = gc.collect()
        df = change2float32(df)
        df = self.optimize_ints(df)
        #df.drop('tweet_attachment_class', axis=1, inplace=True)
        # try using 'media_id' columns

        df_media = pd.read_csv(osj(base_dir, 'Tweets',f'train_tweets_vectorized_media.csv'))
        df_media_test = pd.read_csv(osj(base_dir, 'Tweets',f'test_tweets_vectorized_media.csv'))
        df_media = pd.concat([df_media, df_media_test])
        df_media = change2float32(df_media)
        df = merge_df2media(df, df_media)
        del df_media, df_media_test; _ = gc.collect()

        df_text = pd.read_csv(osj(base_dir, 'Tweets',f'train_tweets_vectorized_text.csv'))
        df_text_test = pd.read_csv(osj(base_dir, 'Tweets',f'test_tweets_vectorized_text.csv'))
        df_text = pd.concat([df_text, df_text_test])
        text_feat_cols = ['text_'+ col for col in df_text.columns if col.startswith('feature_')]
        df_text.columns = ['tweet_id'] + text_feat_cols
        df_text.loc[:, text_feat_cols] = np.log(df_text[text_feat_cols] + 13)
        df_text = change2float32(df_text)
        df = df.merge(df_text, how='left', on='tweet_id')
        del df_text, df_text_test; _ = gc.collect()

        users = pd.read_csv(osj(base_dir, 'Users','users.csv'))
        # log of _count feats

        users_des = pd.read_csv(osj(base_dir, 'Users','user_vectorized_descriptions.csv'))
        # for col in ['tweet_hashtag_count','tweet_url_count','tweet_mention_count']:
        #     users[col] = users[col].astype(int)
        users_img = pd.read_csv(osj(base_dir, 'Users','user_vectorized_profile_images.csv'))
        user_des_feat_cols = ['user_des_'+col for col in users_des.columns if col.startswith('feature')]
        users_des.columns = ['user_id'] + user_des_feat_cols
        user_img_feat_cols = ['user_img_'+col for col in users_img.columns if col.startswith('feature')]
        users_img.columns = ['user_id'] + user_img_feat_cols
        # user_data = users  #  .merge(users, how='left', on='user_id')
        user_data = users.merge(users_des, how='left', on='user_id')
        user_data = user_data.merge(users_img, how='left', on='user_id')
        user_data = change2float32(user_data)
        user_data = self.optimize_ints(user_data)        # # no nulls in user_data 25-may
        df = df.merge(user_data, how='left', left_on='tweet_user_id', right_on='user_id')
        df.drop('user_id', axis=1, inplace=True)
        df = tweets_user_created_date(df)  # add feats: number of user tweets in time period (year, month, day, hour)

        # df = add_num_media_user(df)
        del users_des, users_img, user_data;
        _ = gc.collect()

        return df, test_tweet_ids

    def get_data_stage2(self, cfg, df, test_tweet_ids):
        def add_topic_count(df):
            # and drop the column
            nan_replace = '0'
            # if cfg.one_hot_encode:
            topics = df['tweet_topic_ids'].fillna(f'[{nan_replace}]')
            # topics_xnan = train['tweet_topic_ids'].dropna()
            # fill_value = topicsx_xnan.apply(lambda x: len(eval(x))).mean()
            # fill_value = topicx_xnan.apply(lambda x: len(eval(x))).median()
            n_topics = topics.apply(lambda x: len(eval(x)))
            n_topics_mean = n_topics.mean()
            n_topics = np.where(topics==nan_replace, n_topics_mean, n_topics)
            df['n_topics'] = n_topics.astype(int)
            # add topic_{i} columns
            topics_df = df['tweet_topic_ids'].fillna(f'[{nan_replace}]')
            topics_df = topics_df.apply(lambda x: list(np.unique(eval(x))))
            temp_df = pd.DataFrame(topics_df.to_list()).fillna(int(nan_replace)).astype(int)
            topics_ohe = pd.DataFrame(np.zeros((df.shape[0], len(self.unique_topic_ids)), dtype=np.int8),
                                      index=df.index, columns=[f"topic_id_{i}" for i in self.unique_topic_ids])

            for icol in range(len(temp_df.columns)):
                for jrow in range(len(temp_df)):
                    val = int(temp_df.iat[jrow, icol])
                    topics_ohe.loc[:, f'topic_id_{val}'].iloc[jrow] = 1
            df = pd.concat([df, topics_ohe], axis=1)
            df.drop('tweet_topic_ids', axis=1, inplace=True)
            return df

        df = add_date_feats(df, impute_nulls=cfg.impute_nulls)  # # todo reinstate index - recover df sorted by date

        df = bin_feats(df, feats=['tweet_mention_count','user_tweet_count',
                                  'user_followers_count','user_following_count',
                                  'user_listed_on_count'])
        df = add_topic_count(df)

        bool_cols = df.select_dtypes(include='bool').columns
        df[bool_cols] = df[bool_cols].astype(np.int8)

        # cols to categorical:
        # df = to_categorical(df)
        # 'tweet_language_id',

        if cfg.one_hot_encode:
            df = ohe_func(df, cat_col='tweet_language_id', ohe_tfm=LabelBinarizer())
            df = ohe_func(df, cat_col='tweet_attachment_class', ohe_tfm=LabelBinarizer())
        else:
            df['tweet_attachment_class'] = df['tweet_attachment_class'].astype('category').cat.codes
        # df = to_binary_categories(df, cat_col='tweet_language_id')

        media_img_feat_cols = [col for col in df.columns if col.startswith('img_feature_')]
        if cfg.impute_nulls:
            df.loc[:,media_img_feat_cols] = df[media_img_feat_cols].fillna(self.impute_img_feature_nulls)
        if cfg.add_user_virality:
            df = self.add_virality_feature(df, test_tweet_ids)

        df = freq_encoding(df, freq_cols=['tweet_user_id'], main_col='tweet_id')

        cols_drop = ['tweet_created_at_year', 'tweet_created_at_month',
                     'tweet_created_at_day',
                     'days_since_user', 'user_created_at_year', 'user_created_at_month',
                     'user_verified', 'user_has_url']
        if cfg.one_hot_encode:
            lang_leave_ids = [0, 1, 3]
            cols_drop += [f'tweet_language_id_{i}' for i in range(31)
                          if i not in lang_leave_ids
                          ]
            for col in cols_drop:
                if col in df.columns:
                    df.drop(col, axis=1, inplace=True)
                    # print(f"Dropped col: {col}")

        # self.media_img_feat_cols = [col for col in df.columns if col.startswith('img_feature')]
        # self.text_feat_cols = [col for col in df.columns if col.startswith('text_feature')]
        # self.user_des_feat_cols = [col for col in df.columns if col.startswith('user_des_feature')]
        # self.user_img_feat_cols = [col for col in df.columns if col.startswith('user_img_feature')]

        # print("df.shape after merging all csv files:", df.shape)
        # print("df.dtypes.value_counts():\n", df.dtypes.value_counts())

        train = df[~df['tweet_id'].isin(test_tweet_ids)]
        test = df[df['tweet_id'].isin(test_tweet_ids)]
        del test['virality']; _ = gc.collect()
        print(f"train.shape = {train.shape}, test.shape = {test.shape}")
        return train, test
        # end of def get_data

    def add_virality_feature(self, df, test_tweet_ids):
        # df_train = df[~df['tweet_id'].isin(test_tweet_ids)]
        df_train = df[~df['virality'].isnull()]
        viral_user = df_train.groupby('tweet_user_id')['virality'].mean().reset_index()
        viral_user.columns = ['tweet_user_id', 'user_virality']
        df = df.merge(viral_user, how='left', on='tweet_user_id')
        return df

    def optimize_ints(self, df):
        int8_candidates = self.cols2int8
        # for col in ['tweet_created_at_year', 'user_created_at_year']:
        #     if col in df.columns:
        #         df.loc[:, col] = df.loc[:, col] - 2000
        #         df.loc[:, col] = df.loc[:, col].astype(np.int8)
        for col in int8_candidates:
            if (col in df.columns) and (df[col].isnull().sum()==0):
                df.loc[:, col] = df.loc[:, col].astype(np.int8)
        return df
    # end of class Features

def add_feats_after_load(base_dir, df, test):
    # df - train
    users = pd.read_csv(osj(base_dir, 'Users', 'users.csv'))
    # add log feats and drop
    user_count_feats = [col for col in users.columns if col.endswith('_count')
                                                    and col in df.columns]
    def log_count_feats(df):
        # df - concat of [train, test]
        users.loc[:, user_count_feats] = np.log(users[user_count_feats] + 2)
        df.drop(user_count_feats, axis=1, inplace=True)
        df = df.merge(users[['user_id']+user_count_feats], how='left',
                                    left_on='tweet_user_id', right_on='user_id')
        user_id_cols_drop = [col for col in df.columns if col in ['user_id', 'user_id_x','user_id_y']]
        df.drop(user_id_cols_drop, axis=1, inplace=True)
        return df

    df = pd.concat([df, test])

    # df = tweets_user_created_date(df)
    df = log_count_feats(df)

    train = df[~df['virality'].isnull()]
    test = df[df['virality'].isnull()]
    del test['virality']
    return train, test


class NormalizeFeats_Parallel():
    """ https://scikit-learn.org/stable/computing/parallelism.html
    from joblib import parallel_backend

    with parallel_backend('threading', n_jobs=2):
    # Your scikit-learn code here

    """
    def __init__(self, feat_cols: list):
        self.feat_cols = feat_cols
        self.scalers_dict = {}
    def normalize_data(self, df,  mode='train', scaler=StandardScaler()):
        if mode =='train':
            for col in self.feat_cols:
                with parallel_backend('threading', n_jobs=n_cpus):
                    scaler.fit(df[col].values.reshape(-1,1))
                    self.scalers_dict[col] = scaler
                    # scaler.fit(df[feat_cols].values)
                    df.loc[:,col] = self.scalers_dict[col].transform(df[col].values.reshape(-1,1))
        else:
            for col in self.feat_cols:
                with parallel_backend('threading', n_jobs=n_cpus):
                    df.loc[:,col] = self.scalers_dict[col].transform(df[col].values.reshape(-1,1))
        return df
    # end of NormalizeFeats class

class NormalizeFeats():
    def __init__(self, feat_cols: list):
        self.feat_cols = feat_cols
        self.scalers_dict = {}
    def normalize_data(self, df,  mode='train', scaler=StandardScaler()):
        if mode =='train':
            for col in self.feat_cols:
                scaler.fit(df[col].values.reshape(-1,1))
                self.scalers_dict[col] = scaler
                # scaler.fit(df[feat_cols].values)
                df.loc[:,col] = self.scalers_dict[col].transform(df[col].values.reshape(-1,1))
        else:
            for col in self.feat_cols:
                df.loc[:,col] = self.scalers_dict[col].transform(df[col].values.reshape(-1,1))
        return df
    # end of NormalizeFeats class


def transform_joint(train, test=None, norm_cols=None, tfm = StandardScaler()):
    # normalize joint train test data in chunks by columns

    l_train = len(train)
    if len(norm_cols) < 1000:
        if isinstance(test, pd.DataFrame):
            assert train[norm_cols].columns.equals(test[norm_cols].columns)
            data = pd.concat([train[norm_cols], test[norm_cols]]).values
        else:
            data = train[norm_cols].values
        with parallel_backend('threading', n_jobs=n_cpus):
            tfm.fit(data)
            data = tfm.transform(data)
        train.loc[:, norm_cols] = data[:l_train]
        if isinstance(test, pd.DataFrame):
            test.loc[:, norm_cols] = data[l_train:]
    else: # len(norm_cols) >= 1000
        all_col_chunks = [norm_cols[i:i+1000] for i in range(0, len(norm_cols), 1000)]
        for cols_chunk in all_col_chunks:
            if isinstance(test, pd.DataFrame):
                assert train[norm_cols].columns.equals(test[norm_cols].columns)
                data_chunk = pd.concat([train[cols_chunk], test[cols_chunk]]).values
            else:
                data_chunk = train[cols_chunk]
            scaler = StandardScaler()
            with parallel_backend('threading', n_jobs=n_cpus):
                tfm.fit(data_chunk)
                data_chunk = tfm.transform(data_chunk)
            train.loc[:, cols_chunk] = data_chunk[:l_train] # todo LONGEST RUNTIME and memory
            if isinstance(test, pd.DataFrame):
                test.loc[:, cols_chunk] = data_chunk[l_train:] # todo LONGEST RUNTIME and memory

    return train, test  # test cab be None

def normalize_joint(train, test=None, norm_cols=None):
    # normalize joint train test data in chunks by columns

    l_train = len(train)
    if len(norm_cols) < 1000:
        if isinstance(test, pd.DataFrame):
            assert train[norm_cols].columns.equals(test[norm_cols].columns)
            data = pd.concat([train[norm_cols], test[norm_cols]]).values
        else:
            data = train[norm_cols].values
        scaler = StandardScaler()
        with parallel_backend('threading', n_jobs=n_cpus):
            scaler.fit(data)
            data = scaler.transform(data)
        train.loc[:, norm_cols] = data[:l_train]
        if isinstance(test, pd.DataFrame):
            test.loc[:, norm_cols] = data[l_train:]
    else: # len(norm_cols) >= 1000
        all_col_chunks = [norm_cols[i:i+1000] for i in range(0, len(norm_cols), 1000)]
        for cols_chunk in all_col_chunks:
            if isinstance(test, pd.DataFrame):
                assert train[norm_cols].columns.equals(test[norm_cols].columns)
                data_chunk = pd.concat([train[cols_chunk], test[cols_chunk]]).values
            else:
                data_chunk = train[cols_chunk]
            scaler = StandardScaler()
            with parallel_backend('threading', n_jobs=n_cpus):
                scaler.fit(data_chunk)
                data_chunk = scaler.transform(data_chunk)
            train.loc[:, cols_chunk] = data_chunk[:l_train] # todo LONGEST RUNTIME and memory
            if isinstance(test, pd.DataFrame):
                test.loc[:, cols_chunk] = data_chunk[l_train:] # todo LONGEST RUNTIME and memory

    return train, test  # test cab be None

def normalize_joint_parallel(train, test, norm_cols, num_workers=6):
    # normalize joint train test data in chunks by columns
    from joblib import Parallel, delayed
    l_train = len(train)
    assert train[norm_cols].columns.equals(test[norm_cols].columns)
    if len(norm_cols) < 1000:
        data = pd.concat([train[norm_cols], test[norm_cols]]).values
        scaler = StandardScaler()
        with parallel_backend('threading', n_jobs=n_cpus):
            scaler.fit(data)
            data = scaler.transform(data)
        train.loc[:, norm_cols] = data[:l_train]
        test.loc[:, norm_cols] = data[l_train:]
    else: # len(norm_cols) >= 1000
        all_col_chunks = [norm_cols[i:i+1000] for i in range(0, len(norm_cols), 1000)]

        for cols_chunk in all_col_chunks:
            data_chunk = pd.concat([train[cols_chunk], test[cols_chunk]]).values
            scaler = StandardScaler()
            with parallel_backend('threading', n_jobs=n_cpus):
                scaler.fit(data_chunk)
                data_chunk = scaler.transform(data_chunk)
            train.loc[:, cols_chunk] = data_chunk[:l_train] # todo LONGEST RUNTIME and memory
            test.loc[:, cols_chunk] = data_chunk[l_train:] # todo LONGEST RUNTIME and memory

    return train, test


def split2folds_user_viral(df, n_folds, seed_folds, label_cols=None, foldnum_col='fold'):
    # df is added foldnum_col='fold' column based on KFoldMethod = StratifiedKFold class
    #     applied to label_col='label' column

    temp_col = label_cols[0] + "_" + label_cols[1]
    df[temp_col] = df[label_cols[0]].astype(str) + "_" + df[label_cols[1]].astype(str)
    df[temp_col] =df[temp_col].astype('category')

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed_folds)

    df[foldnum_col] = np.nan
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(df.shape[0]), df[temp_col])):
        df.iloc[val_idx, df.columns.get_loc(foldnum_col)] = fold
    df[foldnum_col] = df[foldnum_col].astype(int)
    # assert df.isnull().sum().sum() == 0, "Error: null values in df"
    del df[temp_col]
    return df

def split2folds_viral_only(df, n_folds, seed_folds, label_col='label', foldnum_col='fold'):
    # df is added foldnum_col='fold' column based on KFoldMethod = StratifiedKFold class
    #     applied to label_col='label' column

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed_folds)

    df[foldnum_col] = np.nan
    for fold, (train_idx, val_idx) in enumerate(skf.split(df.values[:,:1], df[label_col])):
        df.iloc[val_idx, df.columns.get_loc(foldnum_col)] = fold
    df[foldnum_col] = df[foldnum_col].astype(int)
    # assert df.isnull().sum().sum() == 0, "Error: null values in df"
    return df

def split2folds_simple(df, n_folds, seed_folds, foldnum_col='fold'):
    # df is added foldnum_col='fold' column based on KFoldMethod = StratifiedKFold class
    #     applied to label_col='label' column

    skf = KFold(n_splits=n_folds, shuffle=True, random_state=seed_folds)

    df[foldnum_col] = np.nan
    for fold, (train_idx, val_idx) in enumerate(skf.split(df.values[:,:1])):
        df.iloc[val_idx, df.columns.get_loc(foldnum_col)] = fold
    df[foldnum_col] = df[foldnum_col].astype(int)
    # assert df.isnull().sum().sum() == 0, "Error: null values in df"
    return df

def get_feat_cols(train):
    feat_cols = [col for col in train.columns if (col not in ['virality', 'tweet_id',
                                                             'fold','is_test'])
                                                and not col.startswith('target_')]
    media_img_feat_cols = [col for col in train.columns if col.startswith('img_feature')]
    text_feat_cols = [col for col in train.columns if col.startswith('text_feature')]
    user_des_feat_cols = [col for col in train.columns if col.startswith('user_des_feature')]
    user_img_feat_cols = [col for col in train.columns if col.startswith('user_img_feature')]

    feats_some = [col for col in feat_cols if not col in media_img_feat_cols +
                                text_feat_cols + user_img_feat_cols + user_des_feat_cols]
    # print(f"Null values:\n{train[feat_cols].isnull().sum().sort_values(ascending=False).head(2)}")
    return (feat_cols, media_img_feat_cols, text_feat_cols,
            user_des_feat_cols, user_img_feat_cols, feats_some)


def get_folds(cfg, train, default_seed_folds=24):
    # if cfg.seed_folds == default_seed_folds:
    #     folds = pd.read_csv(cfg.folds_split_path)
    #     if 'fold' in train.columns:
    #         del train['fold']
    #     train = train.merge(folds, how='left', on='tweet_id')
    #     train.dropna(0, subset=['fold'], inplace=True)
    # else:
    #     if cfg.folds_split_method == 'user_viral':
    train = split2folds_user_viral(train, cfg.n_folds, cfg.seed_folds,
                                   label_cols=['tweet_user_id', 'virality'], foldnum_col='fold')
    print(f"Folds split with {cfg.seed_folds}")
    return train


def add_new_topic_ids(base_dir, df, df_name='train'):
    if df_name=='train_test':
        df_tweets = pd.read_csv(osj(base_dir, 'Tweets', f'train_tweets.csv'),
                                usecols=['tweet_id', 'tweet_topic_ids']
                                )
        df_tweets_test = pd.read_csv(osj(base_dir, 'Tweets', f'test_tweets.csv'),
                                usecols=['tweet_id', 'tweet_topic_ids']
                                )
        df_tweets = pd.concat([df_tweets, df_tweets_test]).reset_index(drop=True)
        # df_tweets = df.reindex(df.index)
    else:
        df_tweets = pd.read_csv(osj(base_dir, 'Tweets', f'{df_name}_tweets.csv'),
                            usecols=['tweet_id', 'tweet_topic_ids']
                            )
    df_tweets.fillna({'tweet_topic_ids': "['0']"}, inplace=True)
    topic_ids = (
        df_tweets['tweet_topic_ids'].str.strip('[]').str.split('\s*,\s*').explode()
            .str.get_dummies().sum(level=0).add_prefix('topic_id_')
    )
    topic_ids.rename(columns=lambda x: x.replace("'", ""), inplace=True)
    topic_ids['tweet_id'] = df_tweets['tweet_id']
    if 'tweet_topic_ids' in df.columns:
        df.drop('tweet_topic_ids')
    df = df.merge(topic_ids, how='left', on='tweet_id')
    return df, list(topic_ids.columns)

def extract_feats_media_text(df):
    # todo extact_feats_media_text
    # Set the target as well as dependent variables from image data.
    y = vectorized_media_df['virality']
    x = vectorized_media_df.loc[:, vectorized_media_df.columns.str.contains("img_")]

    # Run Lasso regression for feature selection.
    sel_model = SelectFromModel(LogisticRegression(C=1, penalty='l1', solver='liblinear'))

    # time the model fitting
    start = timeit.default_timer()

    # Fit the trained model on our data
    sel_model.fit(x, y)

    stop = timeit.default_timer()
    print('Time: ', stop - start)

    # get index of good features
    sel_index = sel_model.get_support()

    # count the no of columns selected
    counter = collections.Counter(sel_model.get_support())
    print(counter)