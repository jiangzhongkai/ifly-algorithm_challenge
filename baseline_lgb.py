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
from sklearn.model_selection import GridSearchCV,StratifiedKFold,RandomizedSearchCV,train_test_split,cross_val_score,ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectFromModel,RFECV,VarianceThreshold,SelectKBest,chi2,RFE,SelectPercentile
from sklearn.feature_extraction.text import CountVectorizer
import os
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
from feature_enginee import *
import gc
import warnings

warnings.filterwarnings("ignore")
sns.set(style = "whitegrid",color_codes = True)
sns.set(font_scale = 1)


#TODO:数据的加载
def load_params():

    print("loding data.....")
    start = time.time()
    PATH = "Data/"
    allData = pd.read_csv(PATH +"allData.csv")

    trainData_1 = pd.read_csv(PATH+"train_data.txt",delimiter='\t')
    testData_1 = pd.read_csv(PATH+"test_data.txt",delimiter='\t')

    allData = reduce_mem_usage(allData)
    # trainData_1 = reduce_mem_usage(trainData_1)
    # testData_1 = reduce_mem_usage(testData_1)


    all_sid = pd.concat([trainData_1["sid"],testData_1["sid"]],axis=0)
    test_sid = testData_1.pop("sid")
    label = trainData_1.pop("label")

    ######################################整个数据特征以及处理结束###############################################################
    #TODO:数据源的获取
    trainData = allData[:trainData_1.shape[0]]
    testData = allData[trainData_1.shape[0]:]
    trainY = label


    for col in ["idfamd5_3", "android_nuni_apptype", "idfamd5_cnt", "idfamd5_2", "idfamd5_1",
                "cross_adunitshowid_and_dvctype", "cross_mediashowid_and_apptype", "android_nuni_make", "osv_len",
                "cross_adunitshowid_and_apptype","macmd5_cnt","pkgname_2_rank","android_nuni_lan","adunitshowid_1_rank","android_nuni_model","pkgname_2_count",
                "mediashowid_1_rank","adunitshowid_cnt","mediashowid_2_count","pkgname_3_rank","mediashowid_3_count","adunitshowid_1_count","mediashowid_1_count",
                "adunitshowid_3_count","mediashowid_cnt","pkgname_3_count","adunitshowid_2_rank","cross_adunitshowid_and_apptype_cnt","cross_mediashowid_and_apptype_cnt",
                "adunitshowid_3_rank","mediashowid_3_rank","adunitshowid_2_count","mediashowid_2_rank"]:

        del trainData[col]
        del testData[col]

    # TODO:删出部分特征试试  ----2019.8.11  涨了将近3个百分点
    # 分别只取前50，75， 100，125，150，175，200，225，250 进行测试，选最好的特征数量

    all_cate = [('nginxtime', 5198), ('ip', 4713), ('imeimd5', 4590), ('city', 3932), ('model', 3635), ('time_of_min', 3471), ('time_of_sec', 3333),
                ('ratio', 3310), ('big_model', 3007), ('time_of_hour', 2700), ('cross_mediashowid_and_model_cnt', 2608), ('cross_adunitshowid_and_city_cnt', 2410),
                ('cross_mediashowid_and_city_cnt', 2390), ('cross_adunitshowid_and_model_cnt', 2348), ('big_model_rank', 2151), ('cross_adunitshowid_and_osv_cnt', 2129),
                ('macmd5_3', 1930), ('model_medi_nuni', 1841), ('reqrealip', 1808), ('model_adidmd_nuni', 1796), ('model_adid_nuni', 1766), ('ver_cnt', 1736),
                ('city_adid_nuni', 1716), ('cross_mediashowid_and_osv_cnt', 1707), ('macmd5', 1683), ('osv', 1647), ('city_medi_nuni', 1599), ('macmd5_1', 1594),
                ('cross_adunitshowid_and_reqrealip_cnt', 1576), ('city_cnt', 1575), ('osv_cnt', 1564), ('cross_adunitshowid_and_make_cnt', 1544), ('model_cnt', 1543),
                ('adunitshowid_2', 1536), ('city_adidmd_nuni', 1504), ('adunitshowid_3', 1496), ('make', 1464), ('adunitshowid_1', 1434), ('req_ip', 1313),
                ('cross_mediashowid_and_make_cnt', 1295), ('ver', 1288), ('cross_mediashowid_and_city', 1270), ('size', 1175), ('h', 1063),
                ('cross_adunitshowid_and_ntt_cnt', 989), ('cross_mediashowid_and_osv', 961), ('cross_mediashowid_and_ip', 948), ('cross_adunitshowid_and_carrier_cnt', 925),
                ('mediashowid_3', 915), ('adidmd5', 904), ('cross_mediashowid_and_reqrealip_cnt', 902), ('make_adid_nuni', 878), ('make_adidmd_nuni', 864),
                ('adid_nuni_make', 839), ('mediashowid_2', 826), ('mediashowid_1', 810), ('cross_adunitshowid_and_city', 789), ('cross_adidmd5_and_city', 785),
                ('adid_nuni_model', 785), ('cross_adunitshowid_and_province_cnt', 769), ('apptype', 746), ('cross_mediashowid_and_model', 733), ('cross_adidmd5_and_ip', 717),
                ('adid_nuni_osv', 673), ('cross_mediashowid_and_reqrealip', 671), ('cross_adunitshowid_and_ip', 666), ('cross_mediashowid_and_carrier_cnt', 663),
                ('px', 658), ('ip_cnt', 651), ('area', 642), ('cross_adidmd5_and_apptype', 626), ('apptype_cnt', 595), ('ip_adid_nuni', 575), ('cross_mediashowid_and_ntt_cnt', 533),
                ('cross_adunitshowid_and_reqrealip', 514), ('cross_adidmd5_and_reqrealip', 514), ('ip_medi_nuni', 503), ('pkgname_2', 502), ('medi_nuni_make', 495), ('cross_mediashowid_and_make', 492),
                ('adid_nuni_city', 486), ('pkgname_3', 484), ('ip_adidmd_nuni', 480), ('cross_adidmd5_and_model', 473), ('cross_adunitshowid_and_model', 470), ('cross_adunitshowid_and_osv', 468), ('imeimd5_cnt', 457),
                ('pkgname_1', 456), ('openudidmd5', 442), ('medi_nuni_osv', 433), ('cross_mediashowid_and_province_cnt', 428), ('cross_adidmd5_and_city_cnt', 420), ('cross_adidmd5_and_province', 415), ('make_medi_nuni', 403),
                ('cost_time', 397), ('adid_nuni_reqrealip', 382), ('medi_nuni_city', 368), ('cross_adunitshowid_and_make', 367), ('pkgname', 347), ('cross_adunitshowid_and_lan_cnt', 339), ('cross_adidmd5_and_make', 337), ('pkgname_1_rank', 335),
                ('meid_adid_nuni', 326), ('carrier', 326), ('medi_nuni_model', 323), ('cross_mediashowid_and_carrier', 299), ('cross_adunitshowid_and_ntt', 298), ('medi_nuni_reqrealip', 296), ('cross_adidmd5_and_osv', 295), ('w', 287), ('cross_adidmd5_and_dvctype', 283),
                ('cross_adidmd5_and_model_cnt', 278), ('adunitshowid', 276), ('adunitshowid_0_rank', 273), ('cross_mediashowid_and_ip_cnt', 270), ('cross_mediashowid_and_lan_cnt', 268), ('adid_nuni_province', 256), ('cross_mediashowid_and_ntt', 246), ('big_model_count', 245),
                ('ntt', 227), ('ppi', 208), ('make_cnt', 200), ('cross_adunitshowid_and_carrier', 193), ('cross_adidmd5_and_reqrealip_cnt', 191), ('macmd5_0', 178), ('ver_len', 174), ('cross_adunitshowid_and_province', 168), ('carrier_cnt', 165), ('macmd5_0_rank', 158), ('lan_cnt', 153),
                ('cross_adidmd5_and_lan', 148), ('lan', 145), ('adid_nuni_ntt', 142), ('cross_adidmd5_and_ip_cnt', 139), ('mediashowid_0_rank', 134), ('adid_nuni_carrier', 133), ('cross_adidmd5_and_carrier', 131), ('mediashowid', 129), ('ntt_cnt', 125), ('medi_nuni_carrier', 124),
                ('reqrealip_cnt', 120), ('cross_adunitshowid_and_ip_cnt', 120), ('cross_adidmd5_and_ntt', 116), ('medi_nuni_ntt', 101), ('medi_nuni_province', 99), ('cross_mediashowid_and_province', 92), ('cross_mediashowid_and_dvctype_cnt', 86), ('ratio_cat', 86), ('cross_adunitshowid_and_dvctype_cnt', 85),
                ('cross_adidmd5_and_osv_cnt', 81), ('dvctype', 75), ('province', 68), ('cross_adidmd5_and_make_cnt', 64), ('dvctype_cnt', 57), ('adid_nuni_ip', 57), ('adid_nuni_lan', 56), ('android_nuni_ip', 50), ('province_adid_nuni', 50), ('macmd5_3_rank', 44), ('medi_nuni_ip', 43), ('android_nuni_reqrealip', 42),
                ('adunitshowid_0', 42), ('medi_nuni_dvctype', 42), ('cross_adidmd5_and_ntt_cnt', 38), ('adunitshowid_0_count', 37), ('android_nuni_city', 36), ('cross_adidmd5_and_apptype_cnt', 32), ('medi_nuni_lan', 31), ('macmd5_1_rank', 29), ('pkgname_1_count', 28), ('cross_adunitshowid_and_lan', 25),
                ('mediashowid_0', 20), ('orientation', 19), ('openudidmd5_cnt', 19), ('cross_mediashowid_and_lan', 19), ('macmd5_0_count', 18), ('cross_adidmd5_and_province_cnt', 18), ('adid_nuni_dvctype', 16), ('android_nuni_ntt', 14), ('cross_adidmd5_and_dvctype_cnt', 14), ('area_cat', 13),
                ('cross_adidmd5_and_carrier_cnt', 13), ('mediashowid_0_count', 11), ('size_cat', 11), ('cross_adidmd5_and_lan_cnt', 8), ('adidmd5_cnt', 7), ('cross_mediashowid_and_dvctype', 5)]
    top_50_feat = all_cate[:50]
    top_75_feat = all_cate[:75]
    top_100_feat = all_cate[:100]
    top_125_feat = all_cate[:125]
    top_150_feat = all_cate[:150]

    #只取重要性前50的特征
    # for col in all_cate:
    #     if col not in top_50_feat:
    #         del trainData[col[0]]
    #         del testData[col[0]]

    # #只取重要性前75的特征
    # for col in all_cate:
    #     if col not in top_75_feat:
    #         del trainData[col[0]]
    #         del testData[col[0]]
    #
    #只取重要性前100的特征
    # for col in all_cate:
    #     if col not in top_100_feat:
    #         del trainData[col[0]]
    #         del testData[col[0]]
    #
    # for col in all_cate:
    #     if col not in top_125_feat:
    #         del trainData[col[0]]
    #         del testData[col[0]]
    #
    for col in all_cate:
        if col not in top_150_feat:
            del trainData[col[0]]
            del testData[col[0]]

    gc.collect()

    # if os.path.exists("Data/feature/base_train_csr.npz") and True:
    #     print("loading csr .....")
    #     base_train_csr = sparse.load_npz("Data/feature/base_train_csr.npz").tocsr().astype("bool")
    #     base_test_csr = sparse.load_npz("Data/feature/base_test_csr.npz").tocsr().astype("bool")
    # else:
    #     base_train_csr = sparse.csr_matrix((len(trainData), 0))
    #     base_test_csr = sparse.csr_matrix((len(testData), 0))
    #
    #     enc = OneHotEncoder()


    # #利用csr进行构造
    # cv = CountVectorizer(min_df=5)
    # cv.fit(all_sid)
    # train_a = cv.transform(trainData_1["sid"])
    # test_a = cv.transform(test_sid)
    #
    # trainData = sparse.hstack((train_a,trainData),'csr')
    # testData = sparse.hstack((test_a,testData),'csr')

    # try:
    #     #做一个循环遍历找到取得最高值的特征百分点个数
    #     feature_select = SelectPercentile  (chi2, percentile= 98)
    #     feature_select.fit(trainData, trainY)
    #     trainData = feature_select.transform(trainData)
    #     testData = feature_select.transform(testData)
    #     end = time.time()
    #     print("chi2 select finish, it cost {}".format(datetime.timedelta(seconds=(end - start))))
    #
    # except:
    #     raise ValueError("error handle....")

    return trainData, testData, trainY, test_sid

if __name__ == "__main__":

    trainData, testData, trainY, test_sid = load_params()
    trainX, testX = trainData.values, testData.values

    #TODO:2019.8.11 利用xgb来生成叶子的节点
    #采用全部数据集进行训练,不过选择重要性较高的特征叶子节点特征的生成
    feat_importance = [('nginxtime', 5146), ('ip', 4758), ('imeimd5', 4739), ('city', 3998), ('model', 3575), ('time_of_min', 3431), ('time_of_sec', 3360),
                       ('ratio', 3256), ('big_model', 3124), ('time_of_hour', 2650), ('cross_mediashowid_and_model_cnt', 2575), ('cross_adunitshowid_and_model_cnt', 2420),
                       ('cross_mediashowid_and_city_cnt', 2374), ('cross_adunitshowid_and_city_cnt', 2362), ('big_model_rank', 2150), ('cross_adunitshowid_and_osv_cnt', 2028),
                       ('macmd5_3', 1941), ('reqrealip', 1861), ('model_medi_nuni', 1851), ('city_adid_nuni', 1792), ('cross_mediashowid_and_osv_cnt', 1758),
                       ('ver_cnt', 1734), ('model_adidmd_nuni', 1717), ('model_adid_nuni', 1691), ('macmd5_1', 1664), ('osv', 1640), ('macmd5', 1624),
                       ('model_cnt', 1572), ('cross_adunitshowid_and_make_cnt', 1572), ('city_cnt', 1571), ('city_medi_nuni', 1543), ('adunitshowid_1', 1520),
                       ('cross_adunitshowid_and_reqrealip_cnt', 1503), ('adunitshowid_3', 1497), ('adunitshowid_2', 1474), ('osv_cnt', 1468), ('city_adidmd_nuni', 1459),
                       ('make', 1430), ('cross_mediashowid_and_make_cnt', 1316), ('ver', 1311), ('req_ip', 1307), ('cross_mediashowid_and_city', 1283), ('size', 1113),
                       ('h', 1020), ('cross_mediashowid_and_ip', 989), ('cross_mediashowid_and_osv', 973), ('cross_adunitshowid_and_ntt_cnt', 968),
                       ('cross_adunitshowid_and_carrier_cnt', 915), ('adidmd5', 910), ('cross_mediashowid_and_reqrealip_cnt', 905), ('adid_nuni_make', 887),
                       ('make_adidmd_nuni', 862), ('cross_adidmd5_and_city', 847), ('mediashowid_3', 845), ('mediashowid_2', 832), ('cross_adidmd5_and_ip', 820),
                       ('mediashowid_1', 811), ('make_adid_nuni', 809), ('adid_nuni_model', 781), ('cross_mediashowid_and_model', 773), ('apptype', 771),
                       ('cross_adunitshowid_and_city', 750), ('cross_adunitshowid_and_province_cnt', 743), ('adid_nuni_osv', 734), ('px', 705), ('ip_cnt', 674),
                       ('cross_mediashowid_and_carrier_cnt', 656), ('cross_adidmd5_and_apptype', 638), ('cross_mediashowid_and_reqrealip', 634),
                       ('cross_adunitshowid_and_ip', 619), ('apptype_cnt', 589), ('area', 569), ('openudidmd5', 554), ('cross_mediashowid_and_make', 538),
                       ('cross_adidmd5_and_reqrealip', 535), ('ip_adid_nuni', 520), ('cross_adidmd5_and_model', 516), ('cross_adunitshowid_and_reqrealip', 515),
                       ('cross_mediashowid_and_ntt_cnt', 514), ('pkgname_2', 514), ('adid_nuni_city', 508), ('cross_adunitshowid_and_model', 506), ('medi_nuni_make', 498),
                       ('ip_medi_nuni', 490), ('ip_adidmd_nuni', 483), ('cross_adunitshowid_and_osv', 480), ('pkgname_3', 461), ('imeimd5_cnt', 445),
                       ('cross_adidmd5_and_city_cnt', 442), ('make_medi_nuni', 438), ('cost_time', 436), ('medi_nuni_osv', 435), ('pkgname_1', 421),
                       ('cross_mediashowid_and_province_cnt', 408), ('cross_adunitshowid_and_make', 401), ('cross_adidmd5_and_make', 396), ('cross_adidmd5_and_province', 387),
                       ('adid_nuni_reqrealip', 369), ('cross_adidmd5_and_osv', 367), ('pkgname', 352), ('cross_adunitshowid_and_lan_cnt', 327), ('medi_nuni_city', 326),
                       ('medi_nuni_reqrealip', 316), ('pkgname_1_rank', 314), ('medi_nuni_model', 304), ('cross_adunitshowid_and_ntt', 300), ('w', 300)]

    # feat_col = [col[0] for col in feat_importance]
    # no_feat_col = [col for col in trainData.columns if col not in feat_col]
    #
    # clf = XgboostFeature()
    # new_feature,new_test_features = clf.fit_model(X_train = trainData[feat_col].values, y_train= trainY, X_test= testData[feat_col].values)
    # print(new_feature,new_test_features)
    #
    # #然后再将生成的叶子特征与原始特征进行拼接
    # trainX = clf.mergeToOne(trainX,new_feature)
    # testX = clf.mergeToOne(testX, new_test_features)


    # #TODO:模型搭建
    start = time.time()
    model = lgb.LGBMClassifier(boosting_type="gbdt",num_leaves=48, max_depth=-1, learning_rate=0.05,
                               n_estimators=3000, subsample_for_bin=50000,objective="binary",min_split_gain=0, min_child_weight=5, min_child_samples=30, #10
                               subsample=0.8,subsample_freq=1, colsample_bytree=1, reg_alpha=3,reg_lambda=5,
                               feature_fraction= 0.9, bagging_fraction = 0.9,#此次添加的
                               seed= 2019,n_jobs=10,slient=True,num_boost_round=3000)
    n_splits = 7
    random_seed = 2019
    skf = StratifiedKFold(shuffle=True,random_state=random_seed,n_splits=n_splits)
    cv_pred= []
    val_score = []
    for idx, (tra_idx, val_idx) in enumerate(skf.split(trainX, trainY)):
        startTime = time.time()
        print("==================================fold_{}====================================".format(str(idx+1)))
        X_train, Y_train = trainX[tra_idx],trainY[tra_idx]
        X_val, Y_val = trainX[val_idx], trainY[val_idx]
        lgb_model = model.fit(X_train,Y_train,eval_names=["train","valid"],eval_metric=["logloss"],eval_set=[(X_train, Y_train),(X_val,Y_val)],early_stopping_rounds=200)
        print(dict(zip(trainData.columns,lgb_model.feature_importances_)))
        print(lgb_model.feature_importances_)

        #验证集进行验证
        val_pred = lgb_model.predict(X_val,num_iteration=lgb_model.best_iteration_)
        val_score.append(f1_score(Y_val,val_pred))
        print("f1_score:",f1_score(Y_val, val_pred))
        test_pred = lgb_model.predict(testX, num_iteration = lgb_model.best_iteration_).astype(int)
        cv_pred.append(test_pred)
        endTime = time.time()
        print("fold_{} finished in {}".format(str(idx+1), datetime.timedelta(seconds= endTime-startTime)))
    # #     if idx == 0:
    # #         cv_pred = np.array(test_pred).reshape(-1.1)
    # #     else:
    # #         cv_pred = np.hstack((cv_pred,np.array(test_pred).reshape(-1,1)))
    end = time.time()
    print('-'*60)
    print("Training has finished.")
    print("Total training time is {}".format(str(datetime.timedelta(seconds=end-start))))
    print(val_score)
    print("mean f1:",np.mean(val_score))
    print('-'*60)

    submit = []
    for line in np.array(cv_pred).transpose():
        submit.append(np.argmax(np.bincount(line)))
    final_result = pd.DataFrame(columns=["sid","label"])
    final_result["sid"] = list(test_sid.unique())
    final_result["label"] = submit
    final_result.to_csv("submitLGB{0}.csv".format(datetime.datetime.now().strftime("%Y%m%d%H%M")),index = False)
    print(final_result.head())