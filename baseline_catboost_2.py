"""-*- coding: utf-8 -*-
 DateTime   : 2019/8/12 9:18
 Author  : Peter_Bonnie
 FileName    : baseline_catboost_2.py
 Software: PyCharm
"""
import numpy as np
import pandas as pd
from baseline_lgb import load_params
from sklearn.model_selection import StratifiedKFold,TimeSeriesSplit
import gc
import time
import datetime
import catboost as cbt
from sklearn.metrics import f1_score


#TODO:载入数据

trainData, testData, trainY, test_sid = load_params()
trainX, testX = trainData.values, testData.values

#先获取类别特征的索引
cat_list = []
feature_name = [col for col in trainData.columns if col not in ["time_of_hour", "time_of_min","time_of_sec"]]


cat_columns = [col for col in trainData.select_dtypes(object).columns]
# cat_columns = [col for col in trainData.columns if trainDatal[col].dtype == "object"]
print(cat_columns)



for col in cat_columns:
    cat_list.append(feature_name.index(col))

print("cate index :",cat_list)
#TODO:模型搭建
start = time.time()
n_splits = 7
random_state = 2019
skf = StratifiedKFold(n_splits=n_splits, shuffle=False,random_state=random_state)

model = cbt.CatBoostClassifier(iterations=10000, learning_rate=0.05, max_depth=8, task_type='GPU',
                               l2_leaf_reg=8, verbose=10, early_stopping_rounds=3000, eval_metric='F1',cat_features=cat_list)

cv_pred = []
cv_score = []

for index, (tra_idx, val_idx) in enumerate(skf.split(trainX, trainY)):
    start_time = time.time()
    print("==============================fold_{}=========================================".format(str(index+1)))
    X_train, Y_train = trainX[tra_idx], trainY[tra_idx]
    X_val, Y_val = trainX[val_idx], trainY[val_idx]
    cbt_model = model.fit(X_train, Y_train,eval_set=(X_val, Y_val),early_stopping_rounds=3000,cat_features=cat_list)
    print(dict(zip(trainData.columns,cbt_model.feature_importances_)))
    val_pred = cbt_model.predict(X_val)
    print("fold_{0},f1_score:{1}".format(str(index+1), f1_score(Y_val,val_pred)))
    cv_score.append(f1_score(Y_val,val_pred))
    test_pred = cbt_model.predict(testX).astype(int)
    cv_pred.append(test_pred)
    end_time = time.time()
    print("finised in {}".format(datetime.timedelta(seconds=end_time-start_time)))

end = time.time()
print('-'*60)
print("Training has finished.")
print("Total training time is {}".format(str(datetime.timedelta(seconds=end-start))))
print(cv_score)
print("mean f1:",np.mean(cv_score))
print('-'*60)

#提交结果
submit= []

for line in np.array(cv_pred).transpose():
    submit.append(np.argmax(np.bincount(line)))

result = pd.DataFrame(columns=["sid","label"])
result["sid"] = list(test_sid.unique())
result["label"] = submit

result.to_csv("submissionCat{}.csv".format(datetime.datetime.now().strftime("%Y%m%d%H%M")),index=False)

print(result.head())
print(result["label"].value_counts())











