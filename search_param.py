"""-*- coding: utf-8 -*-
 DateTime   : 2019/7/30 10:11
 Author  : Peter_Bonnie
 FileName    : baseline.py
 Software: PyCharm
"""
from __future__ import print_function,division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler,StandardScaler,OneHotEncoder,LabelEncoder
from sklearn.model_selection import GridSearchCV,StratifiedKFold,RandomizedSearchCV,train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,ExtraTreesClassifier,\
    AdaBoostClassifier,VotingClassifier
from sklearn.metrics import f1_score
import time
import datetime
import lightgbm as lgb
from lightgbm import plot_importance
import collections
import xgboost as xgb
from xgboost import plot_importance,to_graphviz
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from data_helper import *
import catboost as cat
import gc
import warnings

warnings.filterwarnings("ignore")

"""
基本操作函数
"""
class Util(object):

    def __init__(self, model, cv_params, trainX, trainY, top_n_features, random_state):

        self.model = model
        self.cv_params = cv_params
        self.trainX = trainX
        self.trainY = trainY
        self.top_n_features = top_n_features
        self.random_state = random_state

    #TODO:选择最佳参数
    def search_best_params(self):

        if self.model == None:
            raise ValueError("model cant be None.")

        if self.cv_params == None or not isinstance(self.cv_params,dict):
            raise TypeError("the type of cv_params should be dict.")

        grid = GridSearchCV(estimator = self.model,param_grid = self.cv_params,scoring = "f1",n_jobs = -1,iid = True,cv = 3, verbose= 2)
        grid.fit(self.trainX, self.trainY)

        return grid.best_estimator_, grid.best_params_, grid.best_score_

    #TODO:利用多个模型来选择最佳特征
    def get_top_n_features(self):

        # 随机森林
        rf_est = RandomForestClassifier(self.random_state)
        rf_param_grid = {'n_estimators': [500], 'min_samples_split': [2, 3], 'max_depth': [20]}
        rf_grid = GridSearchCV(rf_est, rf_param_grid, n_jobs=25, cv=10, verbose=1)
        rf_grid.fit(self.trainX, self.trainY)
        # 将feature按Importance排序
        feature_imp_sorted_rf = pd.DataFrame({'feature': list(self.trainX),
                                              'importance': rf_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
        features_top_n_rf = feature_imp_sorted_rf.head(self.top_n_features)['feature']
        print('Sample 25 Features from RF Classifier')
        print(str(features_top_n_rf[:60]))

        # AdaBoost
        ada_est = AdaBoostClassifier(self.random_state)
        ada_param_grid = {'n_estimators': [500], 'learning_rate': [0.5, 0.6]}
        ada_grid = GridSearchCV(ada_est, ada_param_grid, n_jobs=25, cv=10, verbose=1)
        ada_grid.fit(self.trainX, self.trainY)
        # 排序
        feature_imp_sorted_ada = pd.DataFrame({'feature': list(self.trainX),
                                               'importance': ada_grid.best_estimator_.feature_importances_}).sort_values(
            'importance', ascending=False)
        features_top_n_ada = feature_imp_sorted_ada.head(self.top_n_features)['feature']

        # ExtraTree
        et_est = ExtraTreesClassifier(self.random_state)
        et_param_grid = {'n_estimators': [500], 'min_samples_split': [3, 4], 'max_depth': [15]}
        et_grid = GridSearchCV(et_est, et_param_grid, n_jobs=25, cv=10, verbose=1)
        et_grid.fit(self.trainX, self.trainY)
        # 排序
        feature_imp_sorted_et = pd.DataFrame({'feature': list(self.trainX),
                                              'importance': et_grid.best_estimator_.feature_importances_}).sort_values(
            'importance', ascending=False)
        features_top_n_et = feature_imp_sorted_et.head(self.top_n_features)['feature']
        print('Sample 25 Features from ET Classifier:')
        print(str(features_top_n_et[:60]))

        # 将三个模型挑选出来的前features_top_n_et合并
        features_top_n = pd.concat([features_top_n_rf, features_top_n_ada, features_top_n_et],
                                   ignore_index=True).drop_duplicates()

        return features_top_n


    #TODO:多个模型进行投票
    def Vote_Model(self):

        #随机森林
        rf_est = RandomForestClassifier(n_estimators = 750, criterion = 'gini', max_features = 'sqrt',
                                                 max_depth = 3, min_samples_split = 4, min_samples_leaf = 2,
                                                 n_jobs = 50, random_state = 42, verbose = 1)
        #梯度增强
        gbm_est = GradientBoostingClassifier(n_estimators=900, learning_rate=0.0008, loss='exponential',
                                                      min_samples_split=3, min_samples_leaf=2, max_features='sqrt',
                                                      max_depth=3, random_state=42, verbose=1)
        #extraTree
        et_est = ExtraTreesClassifier(n_estimators=750, max_features='sqrt', max_depth=35, n_jobs=50,
                                               criterion='entropy', random_state=42, verbose=1)

        #lgb
        lgb_est = lgb.LGBMClassifier(boosting_type="gbdt",num_leaves=48, max_depth=-1, learning_rate=0.05, n_estimators=3000,\
                                     subsample_for_bin=50000,objective="binary",min_split_gain=0, min_child_weight=5, min_child_samples=10,\
                                     subsample=0.8,subsample_freq=1, colsample_bytree=1, reg_alpha=3,reg_lambda=5, seed= 2019,n_jobs=10,slient=True,num_boost_round=3000)

        #xgb
        # xgb_est = xgb.XGBClassifier()

        #融合模型
        voting_est = VotingClassifier(estimators = [('rf', rf_est),('gbm', gbm_est),('et', et_est),('lgb',lgb_est)],
                                           voting = 'soft', weights = [3,1.5,1.5,4],
                                           n_jobs = 50)
        voting_est.fit(self.trainX, self.trainY)

        return voting_est

#利用stacking来进行融合
class Ensemble(object):

    def __init__(self, n_splits, stacker, base_models):
        """
        :param n_splits: 交叉选择的次数
        :param stacker: 最终融合模型
        :param base_models: 基模型
        """
        self.n_splits = n_splits
        self.stacker = stacker
        self.base_models = base_models
        self.local_name = locals()  #主要是用来生成动态变量

    def fir_predict(self, X, y, T):
        """
        :param X:  training X set
        :param y:  training y set
        :param T:  testing X set
        :return:
        """
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        folds = list(StratifiedKFold(n_splits = self.n_splits, shuffle = True, random_state = 2019).split(X,y))
        S_train = np.zeros((X.shape[0],len(self.base_models)))
        S_test = np.zeros((T.shape[0],len(self.base_models)))

        for i ,clf in enumerate(self.base_models):
            self.local_name['S_test_%s'%i]= np.zeros((T.shape[0], self.n_splits))

            for j, (train_idx,test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                y_holdout = y[test_idx]
                print("Fit Model %d fold %d"%(i,j))
                clf.fit(X_train,y_train)
                y_pred = clf.predict(X_holdout)
                #查看下每一个模型在每一折的f1分数
                print("fold_{0}  f1_score is :{1}".format(str(i),f1_score(y_holdout, y_pred)))
                S_train[test_idx,i] = y_pred
                self.local_name['S_test_%s'%i][:,j] = clf.predict(T)
                print(self.local_name['S_test_%s'%i][:j])
            #这里进行投票的原则来选择
            temp_res = []
            for line in self.local_name['S_test_%s'%i]:
                temp_res.append(np.argmax(np.bincount(line)))
            S_test[:,i] = temp_res
            del temp_res

        #训练第二层模型并进行预测
        self.stacker.fit(S_train, y)
        res = self.stacker.predict(S_test)
        return res

#降低内存消耗
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64','object']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            elif str(col_type)[:5] == 'float':
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
            # else:
            #     if len(df[col].unique()) / len(df[col]) < 0.05:
            #         df[col] = df[col].astype("category")
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df



if __name__ == "__main__":

    trainData = pd.read_csv("Data/train_data.txt",delimiter='\t')
    trainData = my_fillna(trainData)

    trainData.pop("os")
    sid = trainData.pop("sid")
    label = trainData.pop("label")

    # 先对连续特征进行归一化等操作
    continue_feats = ["h", "w", "ppi", "nginxtime"]
    # 需要进行类别编码的特征
    label_encoder_feats = ["ver", "lan", "pkgname", "adunitshowid", "mediashowid", "apptype", "city", "province", "ip",
                           "reqrealip", "adidmd5", "imeimd5", "idfamd5", "openudidmd5", "macmd5", "model", "make", "osv"]
    # 需要进行独热编码的特征
    cate_feats = ["dvctype", "ntt", "carrier"]

    # labelencoder转化
    for fe in ["pkgname", "adunitshowid", "mediashowid", "apptype", "province", "ip", "reqrealip", "adidmd5", "imeimd5",
               "idfamd5", "openudidmd5", "macmd5", "ver", "lan"]:
        le_feat = LabelEncoder()
        le_feat.fit(trainData[fe])
        trainData[fe] = le_feat.transform(trainData[fe])

    # 由于city在类别编码时出现错误，所以我们就手动进行编码
    city_2_idx = dict(zip(list(set(trainData["city"].unique())), range(len(list(set(trainData["city"].unique()))))))

    # 对model, make,osv 自定义编码方式，因为用类别编码的时候，出现了报错，主要是由于取值中含有不可识别字符，后期再进行处理一下
    model_2_idx = dict(zip(list(set(trainData["model"].unique())), range(len(list(set(trainData["model"].unique()))))))
    make_2_idx = dict(zip(list(set(trainData["make"].unique())), range(len(list(set(trainData["make"].unique()))))))
    osv_2_idx = dict(zip(list(set(trainData["osv"].unique())), range(len(list(set(trainData["osv"].unique()))))))

    trainData["city"] = trainData["city"].map(city_2_idx)
    trainData["model"] = trainData["model"].map(model_2_idx)
    trainData["make"] = trainData["make"].map(make_2_idx)
    trainData["osv"] = trainData["osv"].map(osv_2_idx)

    # #对连续变量进行简单的归一化处理
    for fe in continue_feats:
        temp_data = np.reshape(trainData[fe].values, [-1, 1])
        mm = MinMaxScaler()
        mm.fit(temp_data)
        trainData[fe] = mm.transform(temp_data)

    # 对运营商，网络类型进行简单预处理
    trainData["carrier"] = trainData["carrier"].map({0.0: 0, -1.0: 0, 46000.0: 1, 46001.0: 2, 46003.0: 3}).astype(int)
    trainData["ntt"] = trainData["ntt"].astype(int)
    trainData["orientation"] = trainData["orientation"].astype(int)
    trainData["dvctype"] = trainData["dvctype"].astype(int)


    trainX = trainData.values
    trainY = label

    cv_params = {
        "max_depth" : [i for i in range(3,11)],
        "n_estimators" : [i for i in range(1000,4001, 200)],
        "learning_rate":[0.1,0.2,0.01,0.02,0.03,0.04,0.05,0.001,0.002,0.003,0.004,0.005],
        "num_leaves" : [i for i in range(30, 128,8)]
    }

    model = lgb.LGBMClassifier(boosting_type="gbdt",learning_rate=0.05,n_estimators=3000, subsample_for_bin=50000,objective="binary",min_split_gain=0, min_child_weight=5, min_child_samples=10, subsample=0.8,subsample_freq=1, colsample_bytree=1, reg_alpha=3,reg_lambda=5, seed= 2019,n_jobs=10,slient=True,num_boost_round=3000)
    util = Util(model, cv_params=cv_params, trainX=trainX,trainY=trainY,top_n_features=None,random_state=None)
    print(util.search_best_params())

