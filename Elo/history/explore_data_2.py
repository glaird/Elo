# -*- coding: utf-8 -*-
"""
Created on Sat Dec 08 14:50:50 2018

@author: Garrett
"""

import pandas as pd
import seaborn as sns
#import matplotlib as plt
from sklearn.ensemble import RandomForestRegressor
#import xgboost as xgb

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
#df_first_tran = df_history[['card_id','purchase_date']].groupby('card_id').min()
#df_last_tran = df_history[['card_id','purchase_date']].groupby('card_id').max()

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

print df_train.target.mean()
print df_train[['feature_1','target']].groupby('feature_1').mean()
print df_train[['feature_2','target']].groupby('feature_2').mean()
print df_train[['feature_3','target']].groupby('feature_3').mean()
print df_train[['first_active_month','target']].groupby('first_active_month').mean()

sns.distplot(df_train.target)

df_temp1 = pd.merge(df_train, df_unique_merch_count, how='left', on='card_id')
df_temp2 = pd.merge(df_temp1, df_tran_count, how='left', on='card_id')
df_temp3 = pd.merge(df_temp2, unstacked_cat_1, how='left', on='card_id')
df_temp4 = pd.merge(df_temp3, unstacked_cat_2, how='left', on='card_id')
df_final_train = pd.merge(df_temp4, unstacked_cat_3, how='left', on='card_id')
df_final_train.fillna(value=0, inplace=True)

print 'Training data set built'

rfr = RandomForestRegressor()
features = ['feature_1','feature_2','feature_3','unique_merchant_count','total_transaction_count',
            'cat_1_n','cat_1_y','cat_2_1','cat_2_2','cat_2_3','cat_2_4','cat_2_5',
            'cat_3_a','cat_3_b','cat_3_c']
rfr.fit(df_final_train[features], df_final_train.target)

print 'Model Trained'

print 'R^2: ' + str(rfr.score(df_final_train[features], df_final_train['target']))

importance = rfr.feature_importances_
for i in range(0, len(features)):
    print str(i) + '. ' + features[i] + '\t' + str(importance[i])

df_temp1 = pd.merge(df_test, df_unique_merch_count, how='left', on='card_id')
df_temp2 = pd.merge(df_temp1, df_tran_count, how='left', on='card_id')
df_temp3 = pd.merge(df_temp2, unstacked_cat_1, how='left', on='card_id')
df_temp4 = pd.merge(df_temp3, unstacked_cat_2, how='left', on='card_id')
df_final_test = pd.merge(df_temp4, unstacked_cat_3, how='left', on='card_id')
df_final_test.fillna(value=0, inplace=True)

preds = rfr.predict(df_final_test[features])
print 'Scores predicted'
sns.distplot(preds)

df_output = df_test.copy()
df_output['target'] = preds
df_output[['card_id','target']].to_csv("C:/Users/Garrett/Documents/Kaggle/Elo/submissions/submission_2.csv", index=False)
