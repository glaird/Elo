# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 16:02:25 2018

@author: Garrett
"""


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split, KFold
from sklearn import metrics

#import xgboost as xgb


def k_fold_cross_val(folds, estimators, start_estimator, X, y, min_split=2, max_f='auto', model='regressor', metric='mse'):
    count = 0
    n = len(X)
    kf = KFold(n, n_folds=folds)
    kf_dict = dict([("fold_%s" % i,[]) for i in range(1, folds+1)])
    fold = 0
    for train_index, test_index in kf:
        
        fold += 1
        print "Fold: %s" % fold
        X_train, X_test = X.ix[train_index], X.ix[test_index]
        y_train, y_test = y.ix[train_index], y.ix[test_index]
        print y_train.mean()
        print y_test.mean()
        # Increase degree of linear regression polynomial order
        for n in range(start_estimator, estimators+start_estimator):
            if model=='regressor':
                clf = RandomForestRegressor(n_estimators=n, min_samples_split=min_split, max_features=max_f)
            elif model=='classifier':
                clf = RandomForestClassifier(n_estimators=n, min_samples_split=min_split, max_features=max_f)
            clf.fit(X_train, y_train.ravel())
                # Calculate the test MSE and append to the
                # dictionary of all test curves
            y_pred = clf.predict(X_test)
            df_pred = pd.DataFrame(y_pred)
            df_pred.columns = ['is_attributed']
            print 'Predicted mean'
            print df_pred.is_attributed.mean()
            if metric=='mse':
                test_mse = metrics.mean_squared_error(y_test, y_pred)
                kf_dict["fold_%s" % fold].append(test_mse)
            elif metric=='auc':
                fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
                kf_dict["fold_%s" % fold].append(metrics.auc(fpr, tpr))
            
            if count==0:
                importance = clf.feature_importances_
                importance_dict = {}
                for i in range(0, len(features)):
                    importance_dict[features[i]] = importance[i]
                
                counter = 1
                for key in sorted(importance_dict, key=importance_dict.get, reverse=True):
                    print str(counter) + '. ' + str(key) + '\t' + str(importance_dict[key])
                    counter+=1
            #    importances = clf.feature_importances_
            #    indices = np.argsort(importances)[::-1]
            
                # Print the feature ranking
            #    print("Feature ranking:")
            
            #    for f in range(X.shape[1]):
            #        print("%d. %s (%f)" % (f + 1, X.columns[f], importances[indices[f]]))
                count+=1
            #importances = clf.feature_importances_
            #indices = np.argsort(importances)[::-1]

            # Print the feature ranking
            #print("Feature ranking:")

            #for f in range(X.shape[1]):
                #print("%d. feature %s (%f)" % (f + 1, X.columns[f], importances[indices[f]]))
            # Convert these lists into numpy arrays to perform averaging
        kf_dict["fold_%s" % fold] = np.array(kf_dict["fold_%s" % fold])
    # Create the "average test MSE" series by averaging the 
    # test MSE for each degree of the linear regression model,
    # across each of the k folds.
    kf_dict["avg"] = 0
    for i in range(1, folds+1):
        kf_dict["avg"] += kf_dict["fold_%s" % i]
    kf_dict["avg"] /= float(folds)
    
    return kf_dict
    
def build_final(list_df, key):
    count = 0
    for df in list_df:
        if count==0:
            temp_df = df.copy()
        else:
            temp_df = pd.merge(temp_df, df, how='left', on=key)
        count+=1
    return temp_df


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

df_history_merchants = pd.merge(df_history[['card_id','month_lag','merchant_id']], df_merchants, how='left', on='merchant_id')
print 'History Merchants Built'
df_new_merchants = pd.merge(df_new[['card_id','month_lag','merchant_id']], df_merchants, how='left', on='merchant_id')
print 'New Merchants Built'

train_dfs = [df_train]
test_dfs = [df_test]
train_features = list(features)
test_features = list(features)

s_unique_merch_count = df_history[['card_id','merchant_id']].groupby('card_id').merchant_id.nunique()
df_unique_merch_count = s_unique_merch_count.to_frame()
df_unique_merch_count.columns = ['unique_merchant_count']
df_unique_merch_count.reset_index(inplace=True)
train_dfs.append(df_unique_merch_count)
test_dfs.append(df_unique_merch_count)
print 'Unique Merchant Counts Computed'
df_tran_count = df_history[['card_id','merchant_id']].groupby('card_id').count()
df_tran_count.columns = ['total_transaction_count']
df_tran_count.reset_index(inplace=True)
train_dfs.append(df_tran_count)
test_dfs.append(df_tran_count)
print 'Transaction Counts Computed'

df_cat_1_count = df_history[['card_id','category_1','merchant_id']].groupby(['card_id','category_1']).count()
unstacked_cat_1 = df_cat_1_count.unstack(level=-1)
unstacked_cat_1.columns = ['cat_1_n','cat_1_y']
unstacked_cat_1.reset_index(inplace=True)
train_dfs.append(unstacked_cat_1)
test_dfs.append(unstacked_cat_1)
print 'Category 1 variables built'

df_cat_2_count = df_history[['card_id','category_2','merchant_id']].groupby(['card_id','category_2']).count()
unstacked_cat_2 = df_cat_2_count.unstack(level=-1)
unstacked_cat_2.columns = ['cat_2_1','cat_2_2','cat_2_3','cat_2_4','cat_2_5']
unstacked_cat_2.reset_index(inplace=True)
train_dfs.append(unstacked_cat_2)
test_dfs.append(unstacked_cat_2)
print 'Category 2 variables built'

df_cat_3_count = df_history[['card_id','category_3','merchant_id']].groupby(['card_id','category_3']).count()
unstacked_cat_3 = df_cat_3_count.unstack(level=-1)
unstacked_cat_3.columns = ['cat_3_a','cat_3_b','cat_3_c']
unstacked_cat_3.reset_index(inplace=True)
train_dfs.append(unstacked_cat_3)
test_dfs.append(unstacked_cat_3)
print 'Category 3 variables built'

s_unique_merch_count_new = df_new[['card_id','merchant_id']].groupby('card_id').merchant_id.nunique()
df_unique_merch_count_new = s_unique_merch_count_new.to_frame()
df_unique_merch_count_new.columns = ['new_unique_merchant_count']
df_unique_merch_count_new.reset_index(inplace=True)
train_dfs.append(df_unique_merch_count_new)
test_dfs.append(df_unique_merch_count_new)
print 'New Unique Merchant Counts Computed'
df_tran_count_new = df_new[['card_id','merchant_id']].groupby('card_id').count()
df_tran_count_new.columns = ['new_total_transaction_count']
df_tran_count_new.reset_index(inplace=True)
train_dfs.append(df_tran_count_new)
test_dfs.append(df_tran_count_new)
print 'New Transaction Counts Computed'

df_cat_1_count_new = df_new[['card_id','category_1','merchant_id']].groupby(['card_id','category_1']).count()
unstacked_cat_1_new = df_cat_1_count_new.unstack(level=-1)
unstacked_cat_1_new.columns = ['new_cat_1_n','new_cat_1_y']
unstacked_cat_1_new.reset_index(inplace=True)
train_dfs.append(unstacked_cat_1_new)
test_dfs.append(unstacked_cat_1_new)
print 'New Category 1 variables built'

df_cat_2_count_new = df_new[['card_id','category_2','merchant_id']].groupby(['card_id','category_2']).count()
unstacked_cat_2_new = df_cat_2_count_new.unstack(level=-1)
unstacked_cat_2_new.columns = ['new_cat_2_1','new_cat_2_2','new_cat_2_3','new_cat_2_4','new_cat_2_5']
unstacked_cat_2_new.reset_index(inplace=True)
train_dfs.append(unstacked_cat_2_new)
test_dfs.append(unstacked_cat_2_new)
print 'New Category 2 variables built'

df_cat_3_count_new = df_new[['card_id','category_3','merchant_id']].groupby(['card_id','category_3']).count()
unstacked_cat_3_new = df_cat_3_count_new.unstack(level=-1)
unstacked_cat_3_new.columns = ['new_cat_3_a','new_cat_3_b','new_cat_3_c']
unstacked_cat_3_new.reset_index(inplace=True)
train_dfs.append(unstacked_cat_3_new)
test_dfs.append(unstacked_cat_3_new)
print 'New Category 3 variables built'

#Max Lags
df_lags_max = df_history[['card_id','month_lag']].groupby('card_id').max()
df_lags_min = df_history[['card_id','month_lag']].groupby('card_id').min()
df_lags_max.columns = ['max_month_lag']
df_lags_min.columns=['min_month_lag']
df_lags_max.reset_index(inplace=True)
df_lags_min.reset_index(inplace=True)
df_lags = pd.merge(df_lags_min, df_lags_max, how='inner', on='card_id')
train_features.append('max_month_lag')
test_features.append('max_month_lag')
train_features.append('min_month_lag')
test_features.append('min_month_lag')
train_dfs.append(df_lags)
test_dfs.append(df_lags)

df_lags_new_max = df_new[['card_id','month_lag']].groupby('card_id').max()
df_lags_new_min = df_new[['card_id','month_lag']].groupby('card_id').min()
df_lags_new_max.columns = ['new_max_month_lag']
df_lags_new_min.columns = ['new_min_month_lag']
df_lags_new_max.reset_index(inplace=True)
df_lags_new_min.reset_index(inplace=True)
df_lags_new = pd.merge(df_lags_new_max, df_lags_new_min, how='inner', on='card_id')
train_features.append('new_max_month_lag')
test_features.append('new_max_month_lag')
train_features.append('new_min_month_lag')
test_features.append('new_min_month_lag')
train_dfs.append(df_lags_new)
test_dfs.append(df_lags_new)

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
train_dfs.append(unstacked_unique_count_month)
test_dfs.append(unstacked_unique_count_month)

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
train_dfs.append(unstacked_count_month)
test_dfs.append(unstacked_count_month)


#New
s_unique_merch_count_month_new = df_new[['card_id','month_lag','merchant_id']].groupby(['card_id','month_lag']).merchant_id.nunique()
df_unique_merch_count_month_new = s_unique_merch_count_month_new.to_frame()
unstacked_unique_count_month_new = df_unique_merch_count_month_new.unstack(level=-1)
unstacked_unique_count_month_new.columns = ['unique_month_count_new_1','unique_month_count_new_2']
unstacked_unique_count_month_new.reset_index(inplace=True)
train_dfs.append(unstacked_unique_count_month_new)
test_dfs.append(unstacked_unique_count_month_new)


df_tran_count_month_new = df_new[['card_id','month_lag','merchant_id']].groupby(['card_id','month_lag']).count()
unstacked_count_month_new = df_tran_count_month_new.unstack(level=-1)
unstacked_count_month_new.columns = ['count_month_count_new_1','count_month_count_new_2']
unstacked_count_month_new.reset_index(inplace=True)
train_dfs.append(unstacked_count_month_new)
test_dfs.append(unstacked_count_month_new)


print 'Month Groups Complete'

# Month and Category Vars
"""
cat1_count_month = df_history[['card_id','month_lag','category_1','merchant_id']].groupby(['card_id','month_lag','category_1']).count()
cat2_count_month = df_history[['card_id','month_lag','category_2','merchant_id']].groupby(['card_id','month_lag','category_2']).count()
cat3_count_month = df_history[['card_id','month_lag','category_3','merchant_id']].groupby(['card_id','month_lag','category_3']).count()

unstacked_cat1_count_month = cat1_count_month.unstack(level=-1)
unstacked_cat1_count_month = unstacked_cat1_count_month.unstack(level=-1)

unstacked_cat1_count_month.columns =    ['cat_month_count_12',
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
train_dfs.append(unstacked_unique_count_month)
test_dfs.append(unstacked_unique_count_month)

#New
new_cat1_count_month = df_new[['card_id','month_lag','category_1','merchant_id']].groupby(['card_id','month_lag','category_1']).count()
new_cat2_count_month = df_new[['card_id','month_lag','category_2','merchant_id']].groupby(['card_id','month_lag','category_3']).count()
new_cat3_count_month = df_new[['card_id','month_lag','category_2','merchant_id']].groupby(['card_id','month_lag','category_3']).count()
train_dfs.append(new_cat1_count_month)
test_dfs.append(new_cat1_count_month)
train_dfs.append(new_cat2_count_month)
test_dfs.append(new_cat2_count_month)
train_dfs.append(new_cat3_count_month)
test_dfs.append(new_cat3_count_month)
"""

#Merchant Groups
merch_group_count = df_history_merchants[['card_id','merchant_group_id','merchant_id']].groupby(['card_id','merchant_group']).count()
new_merch_group_count = df_new_merchants[['card_id','merchant_group_id','merchant_id']].groupby(['card_id','merchant_group']).count()

merch_cat_count = df_history_merchants[['card_id','merchant_category_id','merchant_id']].groupby(['card_id','merchant_group']).count()
new_merch_cat_count = df_new_merchants[['card_id','merchant_category_id','merchant_id']].groupby(['card_id','merchant_group']).count()

merch_sub_count = df_history_merchants[['card_id','merchant_subsector_id','merchant_id']].groupby(['card_id','merchant_group']).count()
new_merch_sub_count = df_new_merchants[['card_id','merchant_subsector_id','merchant_id']].groupby(['card_id','merchant_group']).count()

train_dfs.append(merch_group_count)
train_dfs.append(new_merch_group_count)
train_dfs.append(merch_cat_count)
train_dfs.append(new_merch_cat_count)
train_dfs.append(merch_sub_count)
train_dfs.append(new_merch_sub_count)

test_dfs.append(merch_group_count)
test_dfs.append(new_merch_group_count)
test_dfs.append(merch_cat_count)
test_dfs.append(new_merch_cat_count)
test_dfs.append(merch_sub_count)
test_dfs.append(new_merch_sub_count)
print 'Merchant Groups Built'

#Merchant Months
merch_group_count_month = df_history_merchants[['card_id','merchant_group_id','merchant_id','month_lag']].groupby(['card_id','merchant_group','month_lag']).count()
new_merch_group_count_month = df_new_merchants[['card_id','merchant_group_id','merchant_id','month_lag']].groupby(['card_id','merchant_group','month_lag']).count()

merch_cat_count_month = df_history_merchants[['card_id','merchant_category_id','merchant_id','month_lag']].groupby(['card_id','merchant_group','month_lag']).count()
new_merch_cat_count_month = df_new_merchants[['card_id','merchant_category_id','merchant_id','month_lag']].groupby(['card_id','merchant_group','month_lag']).count()

merch_sub_count_month = df_history_merchants[['card_id','merchant_subsector_id','merchant_id','month_lag']].groupby(['card_id','merchant_group','month_lag']).count()
new_merch_sub_count_month = df_new_merchants[['card_id','merchant_subsector_id','merchant_id','month_lag']].groupby(['card_id','merchant_group','month_lag']).count()

train_dfs.append(merch_group_count_month)
train_dfs.append(new_merch_group_count_month)
train_dfs.append(merch_cat_count_month)
train_dfs.append(new_merch_cat_count_month)
train_dfs.append(merch_sub_count_month)
train_dfs.append(new_merch_sub_count_month)

test_dfs.append(merch_group_count_month)
test_dfs.append(new_merch_group_count_month)
test_dfs.append(merch_cat_count_month)
test_dfs.append(new_merch_cat_count_month)
test_dfs.append(merch_sub_count_month)
test_dfs.append(new_merch_sub_count_month)



df_train_temp = build_final(train_dfs, 'card_id')
#Build additional features
df_train_temp['new_old_unique_ratio'] = df_train_temp.new_unique_merchant_count / df_train_temp.unique_merchant_count
df_train_temp['new_old_count_ratio'] = df_train_temp.new_total_transaction_count/ df_train_temp.total_transaction_count

            
print 'Starting Velocity'
# Velocity
for i in range(5, 13):
    last = 13 - i
    this = 13 - i - 1
    df_train_temp['unique_'+str(this)+'_'+str(last)] = df_train_temp['unique_month_count_'+str(this)] / df_train_temp['unique_month_count_'+str(last)]
    df_train_temp['count_'+str(this)+'_'+str(last)] = df_train_temp['count_month_count_'+str(this)] / df_train_temp['count_month_count_'+str(last)]
    train_features.append('unique_'+str(this)+'_'+str(last))
    train_features.append('count_'+str(this)+'_'+str(last))
for i in range(1, 2):
    last = 3 - i
    this = 3 - i - 1
    df_train_temp['unique_new_'+str(this)+'_'+str(last)] = df_train_temp['unique_month_count_new_'+str(this)] / df_train_temp['unique_month_count_new_'+str(last)]
    df_train_temp['count_new_'+str(this)+'_'+str(last)] = df_train_temp['count_month_count_new_'+str(this)] / df_train_temp['count_month_count_new_'+str(last)]
    train_features.append('unique_new_'+str(this)+'_'+str(last))
    train_features.append('count_new_'+str(this)+'_'+str(last))

df_final_train = df_train_temp.copy()
df_final_train.replace(np.inf, 100, inplace=True)
# Add Unique Merchant Ratios
df_final_train['unique_rate'] = df_final_train.total_transaction_count / df_final_train.unique_merchant_count
df_final_train['new_unique_rate'] = df_final_train.new_total_transaction_count / df_final_train.new_unique_merchant_count

df_final_train.fillna(value=0, inplace=True)

print 'Training data set built'

# Add outlier predictor
df_final_train['binary_target'] = df_final_train.target.map(lambda x: 1 if x < -20 else 0)
#rfc = RandomForestClassifier()
print 'Training Outlier Model'
outlier_dict = k_fold_cross_val(3, 1, 200, df_final_train[features], df_final_train.binary_target, model='classifier', metric='auc')
print outlier_dict

rfc = RandomForestClassifier(n_estimators=200) #, min_samples_split=0.0001)
rfc.fit(df_final_train[features], df_final_train.binary_target)
scores = rfc.predict(df_final_train[features])
fpr, tpr, _ = roc_curve(df_final_train.binary_target, scores)
plot_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % plot_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

df_final_train['outlier'] = scores
print 'Training Outlier Mean: ' + str(df_final_train.outlier.mean())
features.append('outlier')

# Train Model
#test_dict = k_fold_cross_val(3, 1, 200, df_final_train[features], df_final_train.target) #, 1000, 10)
#print test_dict

rfr = RandomForestRegressor(n_estimators=100)
     
rfr.fit(df_final_train[train_features], df_final_train.target)

print 'Model Trained'

print 'R^2: ' + str(rfr.score(df_final_train[train_features], df_final_train['target']))
trained_mse = metrics.mean_squared_error(y_test, y_pred)
print 'Train MSE: ' + str(trained_mse)
importance = rfr.feature_importances_
importance_dict = {}
for i in range(0, len(_train_features)):
    importance_dict[train_features[i]] = importance[i]

counter = 1
for key in sorted(importance_dict, key=importance_dict.get, reverse=True):
    print str(counter) + '. ' + str(key) + '\t' + str(importance_dict[key])
    counter+=1
    
df_test_temp = build_final(test_dfs, 'card_id')

df_test_temp['new_old_unique_ratio'] = df_test_temp.new_unique_merchant_count / df_test_temp.unique_merchant_count
df_test_temp['new_old_count_ratio'] = df_test_temp.new_total_transaction_count/ df_test_temp.total_transaction_count


# Velocity
for i in range(5, 13):
    last = 13 - i
    this = 13 - i - 1
    df_test_temp['unique_'+str(this)+'_'+str(last)] = df_test_temp['unique_month_count_'+str(this)] / df_test_temp['unique_month_count_'+str(last)]
    df_test_temp['count_'+str(this)+'_'+str(last)] = df_test_temp['count_month_count_'+str(this)] / df_test_temp['count_month_count_'+str(last)]

for i in range(1, 2):
    last = 3 - i
    this = 3 - i - 1
    df_test_temp['unique_new_'+str(this)+'_'+str(last)] = df_test_temp['unique_month_count_new_'+str(this)] / df_test_temp['unique_month_count_new_'+str(last)]
    df_test_temp['count_new_'+str(this)+'_'+str(last)] = df_test_temp['count_month_count_new_'+str(this)] / df_test_temp['count_month_count_new_'+str(last)]

    
df_final_test = df_test_temp.copy()
df_final_test.replace(np.inf, 100, inplace=True)
# Add unique ratios
df_final_test['unique_rate'] = df_final_test.total_transaction_count / df_final_test.unique_merchant_count
df_final_test['new_unique_rate'] = df_final_test.new_total_transaction_count / df_final_test.new_unique_merchant_count


df_final_test.fillna(value=0, inplace=True)

# Outlier Predictor

test_scores = rfc.predict(df_final_test[features])
df_final_test['outlier'] = test_scores
print 'Test Outlier Mean: ' + str(df_final_test.outlier.mean())
test_features.append('outlier')

preds = rfr.predict(df_final_test[test_features])
print 'Scores predicted'
sns.distplot(preds)

print 'Training Scores distribution:'
sns.distplot(df_train.target)

trained_preds = rfr.predict(df_final_train[features])
sns.distplot(trained_preds)

df_output = df_test.copy()
df_output['target'] = preds
#df_output['outlier'] = df_final_test.outlier
#df_output['target'] = df_output.apply(lambda row: row['target'] if row['outlier']<0.5 else -33, axis=1)
df_output[['card_id','target']].to_csv("C:/Users/Garrett/Documents/Kaggle/Elo/submissions/submission_17.csv", index=False)

#for el in [0.0001, 0.00001]:
#    outlier_dict = k_fold_cross_val(3, 1, 200, df_final_train[features], df_final_train.binary_target, min_split=el, model='classifier', metric='auc')
#    print outlier_dict