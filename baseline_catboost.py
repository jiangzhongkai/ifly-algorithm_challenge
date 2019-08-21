# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from pandas import DataFrame as DF
import scipy.spatial.distance as dist
import catboost as cbt
import json
from sklearn.metrics import f1_score
import time
import gc
import math
from tqdm import tqdm
from scipy import stats
from sklearn.cluster import KMeans
from six.moves import reduce
from sklearn.pipeline import Pipeline
from search_param import reduce_mem_usage

from decimal import *
import warnings

warnings.filterwarnings('ignore')

file = "Data/"

import json
from sklearn.metrics import f1_score
import time
import gc
import math
from tqdm import tqdm
from scipy import stats

from six.moves import reduce
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from collections import Counter

from datetime import datetime, timedelta

import warnings

warnings.filterwarnings('ignore')

train = pd.read_table(file + "train_data.txt")
test = pd.read_table(file + "test_data.txt")
test_sid = test["sid"]

all_data = train.append(test).reset_index(drop=True)

# 对时间的处理
all_data['time'] = pd.to_datetime(all_data['nginxtime'] * 1e+6) + timedelta(hours=8)
all_data['day'] = all_data['time'].dt.dayofyear
all_data['hour'] = all_data['time'].dt.hour

# 对sid和时间的处理
all_data['sid_time'] = all_data['nginxtime'].apply(lambda x: Decimal(str(x)[4:-2]))
time_min = all_data["sid"].apply(lambda x: x.split("-")).apply(lambda x: Decimal(x[-1][4:]))

nginxtime_mean = all_data["nginxtime"].mean()
all_data["sample_weight"] = (all_data['nginxtime'] / nginxtime_mean).fillna(1)

all_data["sid_time"] = (time_min - all_data["sid_time"]).apply(lambda x: int(x))
all_data["sid_time" + "_count"] = all_data.groupby(["sid_time"])["sid_time"].transform('count')
all_data["sid_time" + "_rank"] = all_data.groupby(["sid_time"])["sid_time"].transform('count').rank(method='min')

all_data["req_ip"] = all_data.groupby("reqrealip")["ip"].transform("count")

# Data Clean
# 全部变成大写，防止oppo 和 OPPO 的出现
all_data['model'].replace('PACM00', "OPPO R15", inplace=True)
all_data['model'].replace('PBAM00', "OPPO A5", inplace=True)
all_data['model'].replace('PBEM00', "OPPO R17", inplace=True)
all_data['model'].replace('PADM00', "OPPO A3", inplace=True)
all_data['model'].replace('PBBM00', "OPPO A7", inplace=True)
all_data['model'].replace('PAAM00', "OPPO R15_1", inplace=True)
all_data['model'].replace('PACT00', "OPPO R15_2", inplace=True)
all_data['model'].replace('PABT00', "OPPO A5_1", inplace=True)
all_data['model'].replace('PBCM10', "OPPO R15x", inplace=True)

for fea in ['model', 'make', 'lan']:
    all_data[fea] = all_data[fea].astype('str')
    all_data[fea] = all_data[fea].map(lambda x: x.upper())

    from urllib.parse import unquote


    def url_clean(x):
        x = unquote(x, 'utf-8').replace('%2B', ' ').replace('%20', ' ').replace('%2F', '/').replace('%3F', '?').replace(
            '%25', '%').replace('%23', '#').replace(".", ' ').replace('??', ' '). \
            replace('%26', ' ').replace("%3D", '=').replace('%22', '').replace('_', ' ').replace('+', ' ').replace('-',
                                                                                                                   ' ').replace(
            '__', ' ').replace('  ', ' ').replace(',', ' ')

        if (x[0] == 'V') & (x[-1] == 'A'):
            return "VIVO {}".format(x)
        elif (x[0] == 'P') & (x[-1] == '0'):
            return "OPPO {}".format(x)
        elif (len(x) == 5) & (x[0] == 'O'):
            return "Smartisan {}".format(x)
        elif ('AL00' in x):
            return "HW {}".format(x)
        else:
            return x


    all_data[fea] = all_data[fea].map(url_clean)

all_data['big_model'] = all_data['model'].map(lambda x: x.split(' ')[0])
all_data['model_equal_make'] = (all_data['big_model'] == all_data['make']).astype(int)


#TODO:添加部分特征  -- 2019.8.14-------------------------------------------------------------------------------------------------------------------
#TODO:统计特征
#nunique计算
# adid_feat_nunique = ["mediashowid","apptype","city","ip","reqrealip","province","model","dvctype","make","ntt","carrier","osv","lan"]
#
# for feat in adid_feat_nunique:
#     gp1 = all_data.groupby("adunitshowid")[feat].nunique().reset_index()
#     gp1.columns = ["adunitshowid","adid_nuni_"+feat]
#     all_data = all_data.merge(gp1, how = "left",on="adunitshowid")
#
# gp2 = all_data.groupby("mediashowid")["adunitshowid"].nunique().reset_index()
# gp2.columns = ["mediashowid","meid_adid_nuni"]
# all_data = all_data.merge(gp2, how = "left", on = "mediashowid")
#
# gp2 = all_data.groupby("city")["adunitshowid"].nunique().reset_index()
# gp2.columns = ["city","city_adid_nuni"]
# all_data = all_data.merge(gp2, how = "left", on = "city")
#
# gp2 = all_data.groupby("province")["adunitshowid"].nunique().reset_index()
# gp2.columns = ["province","province_adid_nuni"]
# all_data = all_data.merge(gp2, how = "left", on = "province")
#
# gp2 = all_data.groupby("ip")["adunitshowid"].nunique().reset_index()
# gp2.columns = ["ip","ip_adid_nuni"]
# allData = all_data.merge(gp2, how = "left", on = "ip")
#
# gp2 = all_data.groupby("model")["adunitshowid"].nunique().reset_index()
# gp2.columns = ["model","model_adid_nuni"]
# all_data = all_data.merge(gp2, how = "left", on = "model")
#
# gp2 = all_data.groupby("make")["adunitshowid"].nunique().reset_index()
# gp2.columns = ["make","make_adid_nuni"]
# all_data = all_data.merge(gp2, how = "left", on = "make")
#
#
# del gp1
# del gp2
# gc.collect()
#
# #根据对外媒体id进行类别计数
# meid_feat_nunique = ["adunitshowid","apptype","city","ip","reqrealip","province","model","dvctype","make","ntt","carrier","osv","lan"]
# for feat in meid_feat_nunique:
#     gp1 = all_data.groupby("mediashowid")[feat].nunique().reset_index()
#     gp1.columns = ["mediashowid","medi_nuni_"+feat]
#     all_data = all_data.merge(gp1, how = "left",on="mediashowid")
# gp2 = all_data.groupby("city")["mediashowid"].nunique().reset_index()
# gp2.columns = ["city","city_medi_nuni"]
# all_data = all_data.merge(gp2, how = "left", on = "city")
#
# gp2 = all_data.groupby("ip")["mediashowid"].nunique().reset_index()
# gp2.columns = ["ip","ip_medi_nuni"]
# all_data = all_data.merge(gp2, how = "left", on = "ip")
#
# gp2 = all_data.groupby("province")["mediashowid"].nunique().reset_index()
# gp2.columns = ["province","province_medi_nuni"]
# all_data = all_data.merge(gp2, how = "left", on = "province")
#
# gp2 = all_data.groupby("model")["mediashowid"].nunique().reset_index()
# gp2.columns = ["model","model_medi_nuni"]
# all_data = all_data.merge(gp2, how = "left", on = "model")
#
# gp2 = all_data.groupby("make")["mediashowid"].nunique().reset_index()
# gp2.columns = ["make","make_medi_nuni"]
# all_data = all_data.merge(gp2, how = "left", on = "make")
#
# del gp1
# del gp2
# gc.collect()
#
# #adidmd5
# adidmd5_feat_nunique = ["apptype","city","ip","reqrealip","province","model","dvctype","make","ntt","carrier","osv","lan"]
# for feat in adidmd5_feat_nunique:
#     gp1 = all_data.groupby("adidmd5")[feat].nunique().reset_index()
#     gp1.columns = ["adidmd5","android_nuni_"+feat]
#     all_data =all_data.merge(gp1, how= "left", on = "adidmd5")
#
#
# gp2 = all_data.groupby("city")["adidmd5"].nunique().reset_index()
# gp2.columns = ["city","city_adidmd_nuni"]
# all_data = all_data.merge(gp2, how = "left", on = "city")
#
# gp2 = all_data.groupby("ip")["adidmd5"].nunique().reset_index()
# gp2.columns = ["ip","ip_adidmd_nuni"]
# all_data = all_data.merge(gp2, how = "left", on = "ip")
#
# gp2 = all_data.groupby("province")["adidmd5"].nunique().reset_index()
# gp2.columns = ["province","province_adidmd_nuni"]
# all_data = all_data.merge(gp2, how = "left", on = "province")
#
# gp2 = all_data.groupby("model")["adidmd5"].nunique().reset_index()
# gp2.columns = ["model","model_adidmd_nuni"]
# all_data = all_data.merge(gp2, how = "left", on = "model")
#
# gp2 = all_data.groupby("make")["adidmd5"].nunique().reset_index()
# gp2.columns = ["make","make_adidmd_nuni"]
# all_data = all_data.merge(gp2, how = "left", on = "make")
#
# del gp1
# del gp2
# gc.collect()


# feat_1 = ["adunitshowid","mediashowid","adidmd5"]
# feat_2 = ["apptype","city","ip","reqrealip","province","model","dvctype","make","ntt","carrier","osv","lan"]
# cross_feat = []
# for fe_1 in feat_1:
#     for fe_2 in feat_2:
#         col_name = "cross_"+fe_1+"_and_"+fe_2
#         cross_feat.append(col_name)
#         all_data[col_name] = all_data[fe_1].astype(str).values + "_" + all_data[fe_2].astype(str).values
#
# #TODO:对交叉特征进行计数  ---  2019.8.7
# for fe in cross_feat:
#     locals()[fe+"_cnt"] = all_data[fe].value_counts().to_dict()
#     all_data[fe+"_cnt"] = all_data[fe].map(locals()[fe+"_cnt"])
#
#
# for fe in cross_feat:
#     le_feat = LabelEncoder()
#     le_feat.fit(all_data[fe])
#     all_data[fe] = le_feat.transform(all_data[fe])

# city_cnt = all_data["city"].value_counts().to_dict()
# all_data["city_cnt"] = all_data["city"].map(city_cnt)
#
# model_cnt = all_data["model"].value_counts().to_dict()
# all_data["model_cnt"] = all_data["model"].map(model_cnt)
#
# make_cnt = all_data["make"].value_counts().to_dict()
# all_data["make_cnt"] = all_data["make"].map(make_cnt)
#
# ip_cnt = all_data["ip"].value_counts().to_dict()
# all_data["ip_cnt"] = all_data["ip"].map(ip_cnt)
#
# reqrealip_cnt = all_data["reqrealip"].value_counts().to_dict()
# all_data["reqrealip_cnt"] = all_data["reqrealip"].map(reqrealip_cnt)
#
# osv_cnt = all_data["osv"].value_counts().to_dict()
# all_data["osv_cnt"] = all_data["osv"].map(osv_cnt)

# #TODO:交叉特征
# feat_1 = ["adunitshowid","mediashowid","adidmd5"]
# feat_2 = ["apptype","city","ip","reqrealip","province","model","dvctype","make","ntt","carrier","osv","lan"]
# cross_feat = []
# for fe_1 in feat_1:
#     for fe_2 in feat_2:
#         col_name = "cross_"+fe_1+"_and_"+fe_2
#         cross_feat.append(col_name)
#         all_data[col_name] = all_data[fe_1].astype(str).values + "_" + all_data[fe_2].astype(str).values

# #TODO:对交叉特征进行计数  ---  2019.8.7
# for fe in cross_feat:
#     locals()[fe+"_cnt"] = all_data[fe].value_counts().to_dict()
#     all_data[fe+"_cnt"] = all_data[fe].map(locals()[fe+"_cnt"])
#
# for fe in cross_feat:
#     le_feat = LabelEncoder()
#     le_feat.fit(all_data[fe])
#     all_data[fe] = le_feat.transform(all_data[fe])

#TODO:----------------------------------------------------------------------------------------------------------------------------------------------------------------------

i = "adunitshowid"
all_data[i + "_0"], all_data[i + "_1"], all_data[i + "_2"], all_data[i + "_3"] = all_data[i].apply(lambda x: x[0: 8]), \
                                                                                 all_data[i].apply(lambda x: x[8: 16]), \
                                                                                 all_data[i].apply(lambda x: x[16: 24]), \
                                                                                 all_data[i].apply(lambda x: x[24:32])
del all_data[i]

i = "pkgname"
all_data[i + "_1"], all_data[i + "_2"], all_data[i + "_3"] = all_data[i].apply(lambda x: x[8: 16]), all_data[i].apply(
    lambda x: x[16: 24]), all_data[i].apply(lambda x: x[24: 32])
del all_data[i]

i = "mediashowid"
all_data[i + "_0"], all_data[i + "_1"], all_data[i + "_2"], all_data[i + "_3"] = all_data[i].apply(lambda x: x[0: 8]), \
                                                                                 all_data[i].apply(lambda x: x[8: 16]), \
                                                                                 all_data[i].apply(lambda x: x[16: 24]), \
                                                                                 all_data[i].apply(lambda x: x[24: 32])
del all_data[i]

i = "idfamd5"
all_data[i + "_1"], all_data[i + "_2"], all_data[i + "_3"] = all_data[i].apply(lambda x: x[8: 16]), all_data[i].apply(
    lambda x: x[16: 24]), all_data[i].apply(lambda x: x[24: 32])
del all_data[i]

i = "macmd5"
all_data[i + "_0"], all_data[i + "_1"], all_data[i + "_3"] = all_data[i].apply(lambda x: x[0: 8]), all_data[i].apply(
    lambda x: x[8: 16]), \
                                                             all_data[i].apply(lambda x: x[24:32])
del all_data[i]

# H,W,PPI
all_data['size'] = (np.sqrt(all_data['h'] ** 2 + all_data['w'] ** 2) / 2.54) / 1000
all_data['ratio'] = all_data['h'] / all_data['w']
all_data['px'] = all_data['ppi'] * all_data['size']
all_data['mj'] = all_data['h'] * all_data['w']

all_data["ver_len"] = all_data["ver"].apply(lambda x: str(x).split(".")).apply(lambda x: len(x))
osv = all_data["osv"].apply(lambda x: str(x).split("."))
all_data["osv_len"] = osv.apply(lambda x: len(x))

all_data["ip"] = all_data["ip"].map(lambda x: ".".join(x.split(".")[:2]))

num_col = ['h', 'w', 'size', 'mj', 'ratio', 'px']
cat_col = [i for i in all_data.select_dtypes(object).columns if (i not in ['sid', 'label'])]
both_col = []

rankNot = ["idfamd5_1", "idfamd5_2", "idfamd5_3", "ver_len"]

countNot = ["idfamd5_1", "idfamd5_2", "idfamd5_3", "macmd5_1", "macmd5_2", "macmd5_3", "ver_len"]
for i in tqdm(cat_col):
    lbl = LabelEncoder()
    #
    if i not in countNot:
        all_data[i + "_count"] = all_data.groupby([i])[i].transform('count')
        both_col.extend([i + "_count"])
    if i not in rankNot:
        all_data[i + "_rank"] = all_data.groupby([i])[i].transform('count').rank(method='min')
        both_col.extend([i + "_rank"])
    all_data[i] = lbl.fit_transform(all_data[i].astype(str))

for i in tqdm(['w', 'ppi', 'ratio']):
    all_data['{}_count'.format(i)] = all_data.groupby(['{}'.format(i)])['sid'].transform('count')
    all_data['{}_rank'.format(i)] = all_data['{}_count'.format(i)].rank(method='min')

class_num = 8
quantile = []
for i in range(class_num + 1):
    quantile.append(all_data["ratio"].quantile(q=i / class_num))

all_data["ratio_cat"] = all_data["ratio"]
for i in range(class_num + 1):
    if i != class_num:
        all_data["ratio_cat"][((all_data["ratio"] < quantile[i + 1]) & (all_data["ratio"] >= quantile[i]))] = i
    else:
        all_data["ratio_cat"][
            ((all_data["ratio"] == quantile[i]))] = i - 1
all_data["ratio_cat"] = lbl.fit_transform(all_data["ratio_cat"].astype(str))

class_num = 10
quantile = []
for i in range(class_num + 1):
    quantile.append(all_data["mj"].quantile(q=i / class_num))

all_data["mj_cat"] = all_data["mj"]
for i in range(class_num + 1):
    if i != class_num:
        all_data["mj_cat"][((all_data["mj"] < quantile[i + 1]) & (all_data["mj"] >= quantile[i]))] = i
    else:
        all_data["mj_cat"][
            ((all_data["mj"] == quantile[i]))] = i - 1
all_data["mj_cat"] = lbl.fit_transform(all_data["mj_cat"].astype(str))

class_num = 10
quantile = []
for i in range(class_num + 1):
    quantile.append(all_data["size"].quantile(q=i / class_num))

all_data["size_cat"] = all_data["size"]
for i in range(class_num + 1):
    if i != class_num:
        all_data["size_cat"][((all_data["size"] < quantile[i + 1]) & (all_data["size"] >= quantile[i]))] = i
    else:
        all_data["size_cat"][
            ((all_data["size"] == quantile[i]))] = i - 1
all_data["size_cat"] = lbl.fit_transform(all_data["size_cat"].astype(str))

all_data["req_ip_std"] = all_data.groupby("reqrealip")["ip"].transform("std")
all_data["req_ip_skew"] = all_data.groupby("reqrealip")["ip"].skew()

#添加的试试----2019.8.14--- 提高了几个百分点
for col in ['mediashowid_2_rank','adunitshowid_3_rank','make_count','mediashowid_3_count','make_rank','model_count','model_rank']:
    del all_data[col]

feature_name = [i for i in all_data.columns if i not in ['sid', 'label', 'time', "sample_weight"]]

#修改类型，降低内存

# all_data=reduce_mem_usage(all_data,verbose=True)
#进行参数寻优，由于内存受限，先随机进行采样，然后再进行参数寻优
# all_data = all_data.sample(n=100000)

from sklearn.metrics import roc_auc_score
#获取训练集和测试集
# X_train = all_data[:train.shape[0]]
# X_test = all_data[train.shape[0]:]

tr_index = ~all_data['label'].isnull()

'''
X = all_data[tr_index].dropna().reset_index(drop=True)
X_train = X[list(set(feature_name))].reset_index(drop=True)
y = X[['label']].reset_index(drop=True).astype(int)
'''


X_train = all_data[tr_index].reset_index(drop=True)
y = all_data[tr_index][['label']].reset_index(drop=True).astype(int)
X_test = all_data[~tr_index].reset_index(drop=True)

print(X_train.shape, X_test.shape)

print(feature_name)
random_seed = 2019
final_pred = []
cv_score = []


# [0,2,8,9,16]'hour'
cate_feature = ["ver_len", "apptype", "city", "province", "dvctype", "ntt", "carrier", "lan", "orientation",
                "make", "model", "os", ]
cat_list = []
for i in cat_col:
    cat_list.append(feature_name.index(i))
cat_list.append(feature_name.index("ver_len"))
cat_list.append(feature_name.index("osv_len"))
cat_list.append(feature_name.index("ratio_cat"))
cat_list.append(feature_name.index("mj_cat"))
cat_list.append(feature_name.index("size_cat"))

# print("===search_params==============")
#
# cv_params = {
#      "learning_rate" : [0.1,0.2,0.3,0.01,0.02,0.03,0.04,0.05],
#      "max_depth" : [3,4,5,6,7,8,9,10,11]
# }
#
#
# cbt_model = cbt.CatBoostClassifier(iterations=900,task_type="GPU",
#                                        l2_leaf_reg=8, verbose=10, early_stopping_rounds=1000, eval_metric='F1',
#                                        cat_features=cat_list, gpu_ram_part=0.8,boosting_type="Plain",max_bin=129)
# grid = GridSearchCV(estimator=cbt_model,param_grid=cv_params,scoring='f1',n_jobs=-1,cv=3)
# grid.fit(X_train.values, y.values)
# print(grid.best_score_)
# print(grid.best_params_)
# print(grid.best_estimator_)
#
# exit()

skf = StratifiedKFold(n_splits=8, random_state=random_seed, shuffle=True)
# cv_pred = np.zeros((X_train.shape[0],))
# test_pred = np.zeros((X_test.shape[0],))
val_score = []
cv_pred = []

start = time.time()
for index, (train_index, test_index) in enumerate(skf.split(X_train, y)):
    start_time = time.time()
    print("==========================fold_{}=============================".format(str(index+1)))
    train_x, val_x, train_y, val_y = X_train[feature_name].iloc[train_index], X_train[feature_name].iloc[test_index], \
                                       y.iloc[train_index], y.iloc[test_index]
    # cbt_model.fit(train_x[feature_name], train_y,eval_set=(test_x[feature_name],test_y))
    cbt_model = cbt.CatBoostClassifier(iterations=800, learning_rate=0.05, max_depth=7, task_type="GPU",
                                       l2_leaf_reg=8, verbose=10, early_stopping_rounds=1000, eval_metric='F1',
                                       cat_features=cat_list, gpu_ram_part=0.7,boosting_type="Plain",max_bin=129)
    cbt_model.fit(train_x[feature_name], train_y, eval_set=(val_x[feature_name], val_y),use_best_model=True)
                  # sample_weight=X_train[["sample_weight"]].iloc[train_index])

    print(dict(zip(X_train.columns,cbt_model.feature_importances_)))

    val_pred = cbt_model.predict(val_x)
    print("f1_score:{}".format(f1_score(val_y,val_pred)))
    val_score.append(f1_score(val_y, val_pred))

    test_pred = cbt_model.predict(X_test[feature_name],verbose=10).astype(int)
    cv_pred.append(test_pred)
    end_time = time.time()
    print("-"*60)
    print("finished in {}".format(timedelta(seconds=end_time-start_time)))
    print('-'*60)

    # test_pred += cbt_model.predict_proba(X_test[feature_name], verbose=10)[:, 1] / 5
    # y_val = cbt_model.predict_proba(test_x[feature_name], verbose=10)
    # print(Counter(np.argmax(y_val, axis=1)))
    # cv_score.append(f1_score(test_y, np.round(y_val[:, 1])))
    # print(cv_score[-1])
end = time.time()
print("-"*100)
print(val_score)
print("mean f1_score : {}".format(np.mean(val_score)))
print("Total training fininshed in {}".format(timedelta(seconds=end-start)))
print("-"*100)

#提交结果
submit= []

for line in np.array(cv_pred).transpose():
    submit.append(np.argmax(np.bincount(line)))
result = pd.DataFrame()
result["sid"] = test["sid"].values.tolist()
result["label"] = submit

result.to_csv("submissionCat{}.csv".format(datetime.now().strftime("%Y%m%d%H%M")),index=False)

print(result.head())
print(result["label"].value_counts())