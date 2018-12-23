# -*- coding: utf-8 -*-
"""
Created on Sat Dec 08 14:50:50 2018

@author: Garrett
"""

import pandas as pd
import seaborn as sns
#import matplotlib as plt
from sklearn.ensemble import RandomForestRegressor
import numpy as np
#import xgboost as xgb

features = ['feature_1','feature_2','feature_3','unique_merchant_count','total_transaction_count',
            'cat_1_n','cat_1_y','cat_2_1','cat_2_2','cat_2_3','cat_2_4','cat_2_5',
            'cat_3_a','cat_3_b','cat_3_c',
            'new_unique_merchant_count','new_total_transaction_count',
            'new_cat_1_n','new_cat_1_y','new_cat_2_1','new_cat_2_2','new_cat_2_3','new_cat_2_4','new_cat_2_5',
            'new_cat_3_a','new_cat_3_b','new_cat_3_c',
            'new_old_unique_ratio','new_old_count_ratio',
            'unique_month_count_13',
            'unique_month_count_12',
            'unique_month_count_11',
            'unique_month_count_10',
            'unique_month_count_9',
            'unique_month_count_8',
            'unique_month_count_7',
            'unique_month_count_6',
            'unique_month_count_5',
            'unique_month_count_4',
            'unique_month_count_3',
            'unique_month_count_2',
            'unique_month_count_1',
            'unique_month_count_0',
            'count_month_count_13',
            'count_month_count_12',
            'count_month_count_11',
            'count_month_count_10',
            'count_month_count_9',
            'count_month_count_8',
            'count_month_count_7',
            'count_month_count_6',
            'count_month_count_5',
            'count_month_count_4',
            'count_month_count_3',
            'count_month_count_2',
            'count_month_count_1',
            'count_month_count_0',
            'unique_month_count_new_1','unique_month_count_new_2',
            'count_month_count_new_1','count_month_count_new_2']

df_history = pd.read_csv("C:\Users\Garrett\Documents\Kaggle\Elo\historical_transactions.csv")
print 'Loaded ' + str(df_history.shape[0]) + ' historical transactions'

df_merchants = pd.read_csv("C:\Users\Garrett\Documents\Kaggle\Elo\merchants.csv")
print 'Loaded ' + str(df_merchants.shape[0]) + ' merchant records'

df_train = pd.read_csv("C:/Users/Garrett/Documents/Kaggle/Elo/train.csv")
print 'Loaded ' + str(df_train.shape[0]) + ' training records'

df_test = pd.read_csv("C:/Users/Garrett/Documents/Kaggle/Elo/test.csv")
print 'Loaded ' + str(df_test.shape[0]) + ' test records'

df_new = pd.read_csv("C:/Users/Garrett/Documents/Kaggle/Elo/new_merchant_transactions.csv")
print 'Loaded ' + str(df_new.shape[0]) + ' new merchant records'

s_unique_merch_count = df_history[['card_id','merchant_id']].groupby('card_id').merchant_id.nunique()
df_unique_merch_count = s_unique_merch_count.to_frame()
df_unique_merch_count.columns = ['unique_merchant_count']
df_unique_merch_count.reset_index(inplace=True)
print 'Unique Merchant Counts Computed'
df_tran_count = df_history[['card_id','merchant_id']].groupby('card_id').count()
df_tran_count.columns = ['total_transaction_count']
df_tran_count.reset_index(inplace=True)
print 'Transaction Counts Computed'

df_cat_1_count = df_history[['card_id','category_1','merchant_id']].groupby(['card_id','category_1']).count()
unstacked_cat_1 = df_cat_1_count.unstack(level=-1)
unstacked_cat_1.columns = ['cat_1_n','cat_1_y']
unstacked_cat_1.reset_index(inplace=True)
print 'Category 1 variables built'

df_cat_2_count = df_history[['card_id','category_2','merchant_id']].groupby(['card_id','category_2']).count()
unstacked_cat_2 = df_cat_2_count.unstack(level=-1)
unstacked_cat_2.columns = ['cat_2_1','cat_2_2','cat_2_3','cat_2_4','cat_2_5']
unstacked_cat_2.reset_index(inplace=True)
print 'Category 2 variables built'

df_cat_3_count = df_history[['card_id','category_3','merchant_id']].groupby(['card_id','category_3']).count()
unstacked_cat_3 = df_cat_3_count.unstack(level=-1)
unstacked_cat_3.columns = ['cat_3_a','cat_3_b','cat_3_c']
unstacked_cat_3.reset_index(inplace=True)
print 'Category 3 variables built'

s_unique_merch_count_new = df_new[['card_id','merchant_id']].groupby('card_id').merchant_id.nunique()
df_unique_merch_count_new = s_unique_merch_count_new.to_frame()
df_unique_merch_count_new.columns = ['new_unique_merchant_count']
df_unique_merch_count_new.reset_index(inplace=True)
print 'New Unique Merchant Counts Computed'
df_tran_count_new = df_new[['card_id','merchant_id']].groupby('card_id').count()
df_tran_count_new.columns = ['new_total_transaction_count']
df_tran_count_new.reset_index(inplace=True)
print 'New Transaction Counts Computed'

df_cat_1_count_new = df_new[['card_id','category_1','merchant_id']].groupby(['card_id','category_1']).count()
unstacked_cat_1_new = df_cat_1_count_new.unstack(level=-1)
unstacked_cat_1_new.columns = ['new_cat_1_n','new_cat_1_y']
unstacked_cat_1_new.reset_index(inplace=True)
print 'New Category 1 variables built'

df_cat_2_count_new = df_new[['card_id','category_2','merchant_id']].groupby(['card_id','category_2']).count()
unstacked_cat_2_new = df_cat_2_count_new.unstack(level=-1)
unstacked_cat_2_new.columns = ['new_cat_2_1','new_cat_2_2','new_cat_2_3','new_cat_2_4','new_cat_2_5']
unstacked_cat_2_new.reset_index(inplace=True)
print 'New Category 2 variables built'

df_cat_3_count_new = df_new[['card_id','category_3','merchant_id']].groupby(['card_id','category_3']).count()
unstacked_cat_3_new = df_cat_3_count_new.unstack(level=-1)
unstacked_cat_3_new.columns = ['new_cat_3_a','new_cat_3_b','new_cat_3_c']
unstacked_cat_3_new.reset_index(inplace=True)
print 'New Category 3 variables built'

#Max Lags
df_lags = df_history[['card_id','month_lag']].groupby('card_id').max()
df_lags.columns = ['max_month_lag']
df_lags.reset_index(inplace=True)
features.append('max_month_lag')

df_lags_new = df_new[['card_id','month_lag']].groupby('card_id').max()
df_lags_new.columns = ['new_max_month_lag']
df_lags_new.reset_index(inplace=True)
features.append('new_max_month_lag')

df_temp1 = pd.merge(df_train, df_unique_merch_count, how='left', on='card_id')
df_temp2 = pd.merge(df_temp1, df_tran_count, how='left', on='card_id')
df_temp3 = pd.merge(df_temp2, unstacked_cat_1, how='left', on='card_id')
df_temp4 = pd.merge(df_temp3, unstacked_cat_2, how='left', on='card_id')
df_temp5 = pd.merge(df_temp4, unstacked_cat_3, how='left', on='card_id')

df_temp6 = pd.merge(df_temp5, df_unique_merch_count_new, how='left', on='card_id')
df_temp7 = pd.merge(df_temp6, df_tran_count_new, how='left', on='card_id')
df_temp8 = pd.merge(df_temp7, unstacked_cat_1_new, how='left', on='card_id')
df_temp9 = pd.merge(df_temp8, unstacked_cat_2_new, how='left', on='card_id')
df_temp10 = pd.merge(df_temp9, unstacked_cat_3_new, how='left', on='card_id')

#Build additional features
df_temp10['new_old_unique_ratio'] = df_temp10.new_unique_merchant_count / df_temp10.unique_merchant_count
df_temp10['new_old_count_ratio'] = df_temp10.new_total_transaction_count/ df_temp10.total_transaction_count

#Month Grid
s_unique_merch_count_month = df_history[['card_id','month_lag','merchant_id']].groupby(['card_id','month_lag']).merchant_id.nunique()
df_unique_merch_count_month = s_unique_merch_count_month.to_frame()
unstacked_unique_count_month = df_unique_merch_count_month.unstack(level=-1)
unstacked_unique_count_month.columns = ['unique_month_count_13',
                                        'unique_month_count_12',
                                        'unique_month_count_11',
                                        'unique_month_count_10',
                                        'unique_month_count_9',
                                        'unique_month_count_8',
                                        'unique_month_count_7',
                                        'unique_month_count_6',
                                        'unique_month_count_5',
                                        'unique_month_count_4',
                                        'unique_month_count_3',
                                        'unique_month_count_2',
                                        'unique_month_count_1',
                                        'unique_month_count_0']
unstacked_unique_count_month.reset_index(inplace=True)

df_tran_count_month = df_history[['card_id','month_lag','merchant_id']].groupby(['card_id','month_lag']).count()
unstacked_count_month = df_tran_count_month.unstack(level=-1)
unstacked_count_month.columns = ['count_month_count_13',
                                        'count_month_count_12',
                                        'count_month_count_11',
                                        'count_month_count_10',
                                        'count_month_count_9',
                                        'count_month_count_8',
                                        'count_month_count_7',
                                        'count_month_count_6',
                                        'count_month_count_5',
                                        'count_month_count_4',
                                        'count_month_count_3',
                                        'count_month_count_2',
                                        'count_month_count_1',
                                        'count_month_count_0']
unstacked_count_month.reset_index(inplace=True)

#New
s_unique_merch_count_month_new = df_new[['card_id','month_lag','merchant_id']].groupby(['card_id','month_lag']).merchant_id.nunique()
df_unique_merch_count_month_new = s_unique_merch_count_month_new.to_frame()
unstacked_unique_count_month_new = df_unique_merch_count_month_new.unstack(level=-1)
unstacked_unique_count_month_new.columns = ['unique_month_count_new_1','unique_month_count_new_2']
unstacked_unique_count_month_new.reset_index(inplace=True)

df_tran_count_month_new = df_new[['card_id','month_lag','merchant_id']].groupby(['card_id','month_lag']).count()
unstacked_count_month_new = df_tran_count_month_new.unstack(level=-1)
unstacked_count_month_new.columns = ['count_month_count_new_1','count_month_count_new_2']
unstacked_count_month_new.reset_index(inplace=True)

print 'Month Groups Complete'

df_temp11 = pd.merge(df_temp10, unstacked_count_month, how='left', on='card_id')
df_temp12 = pd.merge(df_temp11, unstacked_count_month_new, how='left', on='card_id')
df_temp13 = pd.merge(df_temp12, unstacked_unique_count_month, how='left', on='card_id')
df_temp14 = pd.merge(df_temp13, unstacked_unique_count_month_new, how='left', on='card_id')
df_temp14 = pd.merge(df_temp14, df_lags, how='left', on='card_id')
df_temp14 = pd.merge(df_temp14, df_lags_new, how='left', on='card_id')
            
print 'Starting Velocity'
# Velocity
for i in range(5, 13):
    last = 13 - i
    this = 13 - i - 1
    df_temp14['unique_'+str(this)+'_'+str(last)] = df_temp14['unique_month_count_'+str(this)] / df_temp14['unique_month_count_'+str(last)]
    df_temp14['count_'+str(this)+'_'+str(last)] = df_temp14['count_month_count_'+str(this)] / df_temp14['count_month_count_'+str(last)]
    features.append('unique_'+str(this)+'_'+str(last))
    features.append('count_'+str(this)+'_'+str(last))
for i in range(1, 2):
    last = 3 - i
    this = 3 - i - 1
    df_temp14['unique_new_'+str(this)+'_'+str(last)] = df_temp14['unique_month_count_new_'+str(this)] / df_temp14['unique_month_count_new_'+str(last)]
    df_temp14['count_new_'+str(this)+'_'+str(last)] = df_temp14['count_month_count_new_'+str(this)] / df_temp14['count_month_count_new_'+str(last)]
    features.append('unique_new_'+str(this)+'_'+str(last))
    features.append('count_new_'+str(this)+'_'+str(last))

df_final_train = df_temp14.copy()
df_final_train.replace(np.inf, 100, inplace=True)
df_final_train.fillna(value=0, inplace=True)

print 'Training data set built'

rfr = RandomForestRegressor()

            
rfr.fit(df_final_train[features], df_final_train.target)

print 'Model Trained'

print 'R^2: ' + str(rfr.score(df_final_train[features], df_final_train['target']))

importance = rfr.feature_importances_
importance_dict = {}
for i in range(0, len(features)):
    importance_dict[features[i]] = importance[i]

counter = 1
for key in sorted(importance_dict, key=importance_dict.get, reverse=True):
    print str(counter) + '. ' + str(key) + '\t' + str(importance_dict[key])
    counter+=1
    
df_temp1 = pd.merge(df_test, df_unique_merch_count, how='left', on='card_id')
df_temp2 = pd.merge(df_temp1, df_tran_count, how='left', on='card_id')
df_temp3 = pd.merge(df_temp2, unstacked_cat_1, how='left', on='card_id')
df_temp4 = pd.merge(df_temp3, unstacked_cat_2, how='left', on='card_id')
df_temp5 = pd.merge(df_temp4, unstacked_cat_3, how='left', on='card_id')

df_temp6 = pd.merge(df_temp5, df_unique_merch_count_new, how='left', on='card_id')
df_temp7 = pd.merge(df_temp6, df_tran_count_new, how='left', on='card_id')
df_temp8 = pd.merge(df_temp7, unstacked_cat_1_new, how='left', on='card_id')
df_temp9 = pd.merge(df_temp8, unstacked_cat_2_new, how='left', on='card_id')
df_temp10 = pd.merge(df_temp9, unstacked_cat_3_new, how='left', on='card_id')

df_temp10['new_old_unique_ratio'] = df_temp10.new_unique_merchant_count / df_temp10.unique_merchant_count
df_temp10['new_old_count_ratio'] = df_temp10.new_total_transaction_count/ df_temp10.total_transaction_count

df_temp11 = pd.merge(df_temp10, unstacked_count_month, how='left', on='card_id')
df_temp12 = pd.merge(df_temp11, unstacked_count_month_new, how='left', on='card_id')
df_temp13 = pd.merge(df_temp12, unstacked_unique_count_month, how='left', on='card_id')
df_temp14 = pd.merge(df_temp13, unstacked_unique_count_month_new, how='left', on='card_id')
df_temp14 = pd.merge(df_temp14, df_lags, how='left', on='card_id')
df_temp14 = pd.merge(df_temp14, df_lags_new, how='left', on='card_id')

# Velocity
for i in range(5, 13):
    last = 13 - i
    this = 13 - i - 1
    df_temp14['unique_'+str(this)+'_'+str(last)] = df_temp14['unique_month_count_'+str(this)] / df_temp14['unique_month_count_'+str(last)]
    df_temp14['count_'+str(this)+'_'+str(last)] = df_temp14['count_month_count_'+str(this)] / df_temp14['count_month_count_'+str(last)]

for i in range(1, 2):
    last = 3 - i
    this = 3 - i - 1
    df_temp14['unique_new_'+str(this)+'_'+str(last)] = df_temp14['unique_month_count_new_'+str(this)] / df_temp14['unique_month_count_new_'+str(last)]
    df_temp14['count_new_'+str(this)+'_'+str(last)] = df_temp14['count_month_count_new_'+str(this)] / df_temp14['count_month_count_new_'+str(last)]

    
df_final_test = df_temp14.copy()
df_final_test.replace(np.inf, 100, inplace=True)
df_final_test.fillna(value=0, inplace=True)

preds = rfr.predict(df_final_test[features])
print 'Scores predicted'
sns.distplot(preds)

print 'Training Scores distribution:'
sns.distplot(df_train.target)

df_output = df_test.copy()
df_output['target'] = preds
df_output[['card_id','target']].to_csv("C:/Users/Garrett/Documents/Kaggle/Elo/submissions/submission_7.csv", index=False)
