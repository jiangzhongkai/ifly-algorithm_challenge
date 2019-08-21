from __future__ import print_function,division
"""-*- coding: utf-8 -*-
 DateTime   : 2019/7/30 10:11
 Author  : Peter_Bonnie
 FileName    : baseline.py
 Software: PyCharm
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
from sklearn.preprocessing import MinMaxScaler,StandardScaler,OneHotEncoder,LabelEncoder
from sklearn.model_selection import GridSearchCV,StratifiedKFold,RandomizedSearchCV,train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.metrics import f1_score
import time
import datetime
import lightgbm as lgb
from lightgbm import plot_importance
import collections
import xgboost as xgb
from xgboost import plot_importance,to_graphviz
from data_helper import *
from search_param import  Util,reduce_mem_usage
import catboost as cat
import gc
import warnings

warnings.filterwarnings("ignore")
sns.set(style = "whitegrid",color_codes = True)
sns.set(font_scale = 1)

#TODO:数据的加载
PATH = "Data/"
allData = pd.read_csv(PATH +"allData.csv")
trainData = pd.read_csv(PATH+"train_data.txt",delimiter='\t')
testData = pd.read_csv(PATH+"test_data.txt",delimiter='\t')

#降低内存消耗
allData = reduce_mem_usage(allData)
trainData = reduce_mem_usage(trainData)
testData = reduce_mem_usage(testData)

test_sid = testData.pop("sid")
label = trainData.pop("label")

######################################整个数据特征以及处理结束###############################################################
#TODO:数据源的获取
trainData = allData[:trainData.shape[0]]
testData = allData[trainData.shape[0]:]
trainX, trainY, testX = trainData.values,label, testData.values

#TODO:特征重要性的选择
def f1_sco(preds,valid):
    labels = valid.get_label()
    preds = np.argmax(preds.reshape(2, -1), axis=0)
    score_vali = f1_score(y_true=labels, y_pred=preds, average='weighted')  # 改了下f1_score的计算方式
    return 'f1_score', score_vali, True

# #TODO:模型搭建
xgb_params = {
               'booster': 'gbtree',
               'max_depth': 5,
               'subsample': 0.8,
               'colsample_bytree': 0.8,
               'objective':'binary:logistic',
               'eval_metric': 'logloss',
               "learning_rate": 0.05,
               "seed" : 2019,
               "njob" : -1,
               'silent': True,
          }
start = time.time()
model = xgb.XGBClassifier(**xgb_params)
n_splits = 7
random_seed = 2019
skf = StratifiedKFold(shuffle=True,random_state=random_seed,n_splits=n_splits)
cv_pred= []
val_score = []
for idx, (tra_idx, val_idx) in enumerate(skf.split(trainX, trainY)):
    print("==================================fold_{}====================================".format(str(idx+1)))
    X_train, Y_train = trainX[tra_idx],trainY[tra_idx]
    X_val, Y_val = trainX[val_idx], trainY[val_idx]
    dtrain = xgb.DMatrix(X_train,Y_train)
    dval = xgb.DMatrix(X_val, Y_val)
    watchlists = [(dtrain,'dtrain'),(dval,'dval')]
    bst = xgb.train(dtrain=dtrain, num_boost_round=3000, evals=watchlists, early_stopping_rounds=200, \
        verbose_eval=50, params=xgb_params)
    val_pred = bst.predict(xgb.DMatrix(trainX[val_idx]),ntree_limit = bst.best_ntree_limit)
    val_pred = [0 if i < 0.5 else 1 for i in val_pred]
    val_score.append(f1_score(Y_val,val_pred))
    print("f1_score:", f1_score(Y_val, val_pred))
    test_pred = bst.predict(xgb.DMatrix(testX),ntree_limit = bst.best_ntree_limit)
    test_pred = [0 if i < 0.5 else 1 for i in test_pred]
    cv_pred.append(test_pred)
# #     if idx == 0:
# #         cv_pred = np.array(test_pred).reshape(-1.1)
# #     else:
# #         cv_pred = np.hstack((cv_pred,np.array(test_pred).reshape(-1,1)))
end = time.time()
diff = end - start
print(compute_cost(diff))
submit = []
for line in np.array(cv_pred).transpose():
    submit.append(np.argmax(np.bincount(line)))
final_result = pd.DataFrame(columns=["sid","label"])
final_result["sid"] = list(test_sid.unique())
final_result["label"] = submit

final_result.to_csv("submitXGB{0}.csv".format(datetime.datetime.now().strftime("%Y%m%d%H%M")),index = False)
print(val_score)
print("mean f1:",np.mean(val_score))
print(final_result.head())