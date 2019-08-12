"""-*- coding: utf-8 -*-
 DateTime   : 2019/7/29 9:52
 Author  : Peter_Bonnie
 FileName    : feature_enginee.py
 Software: PyCharm
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import  f1_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.feature_selection import SelectKBest, RFE, SelectFromModel, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost.sklearn import XGBClassifier



def modeling_cross_validation(params, X, y, nr_folds=5):
    # oof_preds = np.zeros(X.shape[0])
    # Split data with kfold
    folds = KFold(n_splits=nr_folds, shuffle=False, random_state=4096)

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
        print("fold n°{}".format(fold_ + 1))
        trn_data = lgb.Dataset(X[trn_idx], y[trn_idx])
        val_data = lgb.Dataset(X[val_idx], y[val_idx])

        num_round = 20000
        clf = lgb.train(params, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=1000,
                        early_stopping_rounds=100)
        val_pred = clf.predict(X[val_idx], num_iteration=clf.best_iteration)


    score = f1_score(val_pred, y)
    # score = mean_squared_error(oof_preds, target)

    return score

def featureSelect_CV(train,columns,target):
    init_cols = columns
    params = {'num_leaves': 120,
              'min_data_in_leaf': 30,
              'objective': 'binary',
              'max_depth': -1,
              'learning_rate': 0.05,
              "min_child_samples": 30,
              "boosting": "gbdt",
              "feature_fraction": 0.9,
              "bagging_freq": 1,
              "bagging_fraction": 0.9,
              "bagging_seed": 11,
              "lambda_l1": 0.02,
              "verbosity": -1}
    best_cols = init_cols.copy()
    best_score = modeling_cross_validation(params, train[init_cols].values, target.values, nr_folds=5)
    print("初始CV score: {:<8.8f}".format(best_score))
    save_remove_feat=[] #用于存储被删除的特征
    for f in init_cols:

        best_cols.remove(f)
        score = modeling_cross_validation(params, train[best_cols].values, target.values, nr_folds=5)
        diff = best_score - score
        print('-' * 10)
        if diff > 0.00002:
            print("当前移除特征: {}, CV score: {:<8.8f}, 最佳cv score: {:<8.8f}, 有效果,删除！！".format(f, score, best_score))
            best_score = score
            save_remove_feat.append(f)
        else:
            print("当前移除特征: {}, CV score: {:<8.8f}, 最佳cv score: {:<8.8f}, 没效果,保留！！".format(f, score, best_score))
            best_cols.append(f)
    print('-' * 10)
    print("优化后CV score: {:<8.8f}".format(best_score))

    return best_cols,save_remove_feat

class SelectFeature(object):

    def __init__(self, X, y, columns):

        self.X = X
        self.y = y
        self.cols  = columns
        self.k = 130

    #TODO: 使用随机森林来选择特征
    def SelectFeatureByModel_1(self):
        lr_selector = SelectFromModel(estimator=LogisticRegression(penalty='l1'))
        lr_selector.fit(self.X.values, self.y)
        lr_support = lr_selector.get_support(indices=True)
        _ = lr_selector.get_support()

        save_feat = []
        for i in list(lr_support):
            save_feat.append(self.X.columns[i])

        return save_feat,_

    def SelectFeatureByModel_2(self):
        rf_selector = SelectFromModel(estimator=RandomForestClassifier(n_estimators=100))
        rf_selector.fit(self.X.values, self.y)
        _ = rf_selector.get_support()
        rf_support = rf_selector.get_support(indices=True)

        save_feat = []
        for i in list(rf_support):
            save_feat.append(self.X.columns[i])

        return save_feat, _

    #TODO-2:利用方差来选择特征
    def SelectFeatureByVariance(self):
        pass

    #TODO-3:利用递归特征消除
    def SelectFeatureByRFE(self):
        rfe_selector = RFE(estimator=LogisticRegression(),n_features_to_select=self.k,step=10,verbose=5)
        rfe_selector.fit(self.X.values,self.y)
        rfe_support = rfe_selector.get_support(indices=True)
        _= rfe_selector.get_support()
        save_feat = []
        for i in list(rfe_support):
            save_feat.append(self.X.columns[i])

        return save_feat,_

    #TODO-4:利用卡方检验来进行特征的选择
    def SelectFeatureByK(self):

        """可以根据scores_来查看每个特征的得分，分越高，表示越重要
           或者根据p值，p值越小，表示置信度越高
        """
        chi_selector = SelectKBest(chi2, k = self.k)
        chi_selector.fit(self.X.values, self.y)
        chi_support = chi_selector.get_support(indices=True)  #返回被选择的特征所在的列
        _ = chi_selector.get_support()
        print(chi_support)
        save_feat = []
        #获取对应的特征
        for i in list(chi_support):
            save_feat.append(self.X.columns[i])

        return save_feat,_

    #TODO-5 :皮尔逊相关系数
    def SelectFeatureByPerson(self):
        pass

    #TODO-6: 获取多个方法的交集
    def merge_Multi_Method(self, feat_name, chi_support, rf_support, rfe_support, lr_support):

        feature_selection_df = pd.DataFrame(columns={"features":feat_name, "Chi":chi_support,"rf":rf_support,"rfe":rfe_support,"lr":lr_support})
        feature_selection_df["total"] = np.sum(feature_selection_df,axis=1)
        feature_selection_df = feature_selection_df.sort_values(["total","features"],ascending=False)
        feature_selection_df.index = range(1,len(feature_selection_df)+1)
        feature_selection_df.to_csv("drop_cols.csv")
        return feature_selection_df



class XgboostFeature():
      ##可以传入xgboost的参数,用来生成叶子节点特征
      ##常用传入特征的个数 即树的个数 默认30
      def __init__(self,n_estimators=30,learning_rate =0.3,max_depth=3,min_child_weight=1,gamma=0.3,subsample=0.8,colsample_bytree=0.8,objective= 'binary:logistic',nthread=4,scale_pos_weight=1,reg_alpha=1e-05,reg_lambda=1,seed=27):
          self.n_estimators=n_estimators
          self.learning_rate=learning_rate
          self.max_depth=max_depth
          self.min_child_weight=min_child_weight
          self.gamma=gamma
          self.subsample=subsample
          self.colsample_bytree=colsample_bytree
          self.objective=objective
          self.nthread=nthread
          self.scale_pos_weight=scale_pos_weight
          self.reg_alpha=reg_alpha
          self.reg_lambda=reg_lambda
          self.seed=seed
          print('Xgboost Feature start, new_feature number:',n_estimators)
      def mergeToOne(self,X,X2):
          X3=[]
          for i in range(X.shape[0]):
              tmp=np.array([list(X[i]),list(X2[i])])
              X3.append(list(np.hstack(tmp)))
          X3=np.array(X3)
          return X3
      ##切割训练
      def fit_model_split(self,X_train,y_train,X_test,y_test):
          ##X_train_1用于生成模型  X_train_2用于和新特征组成新训练集合
          X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split(X_train, y_train, test_size=0.6, random_state=0)
          clf = XGBClassifier(
                 learning_rate =self.learning_rate,
                 n_estimators=self.n_estimators,
                 max_depth=self.max_depth,
                 min_child_weight=self.min_child_weight,
                 gamma=self.gamma,
                 subsample=self.subsample,
                 colsample_bytree=self.colsample_bytree,
                 objective= self.objective,
                 nthread=self.nthread,
                 scale_pos_weight=self.scale_pos_weight,
                 reg_alpha=self.reg_alpha,
                 reg_lambda=self.reg_lambda,
                 seed=self.seed)
          clf.fit(X_train_1, y_train_1)
          y_pre= clf.predict(X_train_2)
          y_pro= clf.predict_proba(X_train_2)[:,1]
          print("pred_leaf=T AUC Score : %f" % metrics.roc_auc_score(y_train_2, y_pro))
          print("pred_leaf=T  Accuracy : %.4g" % metrics.accuracy_score(y_train_2, y_pre))
          new_feature= clf.apply(X_train_2)
          X_train_new2=self.mergeToOne(X_train_2,new_feature)
          new_feature_test= clf.apply(X_test)
          X_test_new=self.mergeToOne(X_test,new_feature_test)
          print("Training set of sample size 0.4 fewer than before")
          return X_train_new2,y_train_2,X_test_new,y_test
      ##整体训练
      def fit_model(self,X_train,y_train,X_test):
          clf = XGBClassifier(
                 learning_rate =self.learning_rate,
                 n_estimators=self.n_estimators,
                 max_depth=self.max_depth,
                 min_child_weight=self.min_child_weight,
                 gamma=self.gamma,
                 subsample=self.subsample,
                 colsample_bytree=self.colsample_bytree,
                 objective= self.objective,
                 nthread=self.nthread,
                 scale_pos_weight=self.scale_pos_weight,
                 reg_alpha=self.reg_alpha,
                 reg_lambda=self.reg_lambda,
                 seed=self.seed)
          clf.fit(X_train, y_train)
          # y_pre= clf.predict(X_test)
          # y_pro= clf.predict_proba(X_test)[:,1]
          # print("pred_leaf=T  AUC Score : %f" % metrics.roc_auc_score(y_test, y_pro))
          # print("pred_leaf=T  Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pre))
          new_feature= clf.apply(X_train)
          X_train_new=self.mergeToOne(X_train,new_feature)
          new_feature_test= clf.apply(X_test)
          X_test_new=self.mergeToOne(X_test,new_feature_test)
          print("Training set sample number remains the same")
          return X_train_new,y_train,X_test_new




