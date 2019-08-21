"""-*- coding: utf-8 -*-
 DateTime   : 2019/8/5 22:06
 Author  : Peter_Bonnie
 FileName    : data_load.py
 Software: PyCharm
"""
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
import re
from data_helper import *
import time
import datetime
import gc
from scipy import sparse
from multiprocessing import cpu_count
import tqdm
import warnings
warnings.filterwarnings("ignore")

print("data procesing.....")
start = time.time()
#TODO:数据的加载
PATH = "Data/"
trainData = pd.read_csv(PATH +"train_data.txt", delimiter= '\t')
testData = pd.read_csv(PATH + "test_data.txt", delimiter= '\t')

# trainData = reduce_mem_usage(trainData, verbose = True)
# testData = reduce_mem_usage(testData, verbose = True)

#TODO: 对osv,make,以及model 进行了比较细致的划分处理  ----修改时间2019.8.3

#osv
trainData["osv"].fillna('0',inplace = True)
trainData["osv"] = trainData["osv"].map(lambda x: re.sub('[a-zA-Z]+','',x))
trainData["osv"] = trainData["osv"].map(lambda x: re.sub('_','.',x))
trainData["osv"] = trainData["osv"].map(lambda x:padding_empty_osv(x))
trainData["osv"] = trainData["osv"].map(lambda x:remove_outlier_osv(x))

testData["osv"].fillna('0', inplace = True)
testData["osv"] = testData["osv"].map(lambda x:re.sub('[a-zA-Z]+','',x))
testData["osv"] = testData["osv"].map(lambda x:re.sub('_','.',x))
testData["osv"] = testData["osv"].map(lambda x:padding_empty_osv(x))
testData["osv"] = testData["osv"].map(lambda x:remove_outlier_osv(x))


#缺失值填充
trainData = my_fillna(trainData)
testData = my_fillna(testData)
######################################3#######################数据处理#######################################################
#获取会话开始时间
trainData["begin_time"] = trainData["sid"].apply(lambda x:int(x.split('-')[-1]))
testData["begin_time"] = testData["sid"].apply(lambda x:int(x.split('-')[-1]))
#获取消耗的时间
trainData["cost_time"] = trainData["nginxtime"] - trainData["begin_time"]
testData["cost_time"] = testData["nginxtime"] - testData["begin_time"]
#将时间戳转化为时分秒的形式

del trainData["begin_time"]
del testData["begin_time"]

#TODO:2019.8.9号添加
trainData["time"] = pd.to_datetime(trainData["nginxtime"], unit='ms')
testData["time"] = pd.to_datetime(testData["nginxtime"], unit='ms')

trainData["time_of_hour"] = trainData["time"].dt.hour.astype(int)
testData["time_of_hour"] = testData["time"].dt.hour.astype(int)
trainData["time_of_min"] = trainData["time"].dt.minute.astype(int)
testData["time_of_min"] = testData['time'].dt.minute.astype(int)
trainData["time_of_sec"] = trainData["time"].dt.second.astype(int)
testData["time_of_sec"] = testData["time"].dt.second.astype(int)

del trainData["time"]
del testData["time"]

#carrier
trainData.carrier[(trainData.carrier == 0) | (trainData.carrier == -1)] = 0
testData.carrier[(testData.carrier == 0) | (testData.carrier == -1)] =0

#ntt
trainData.ntt[(trainData.ntt == 0) | (trainData.ntt == 7)] = 0
testData.ntt[(testData.ntt == 0) | (testData.ntt == 7)] = 0

trainData.ntt[(trainData.ntt == 1) | (trainData.ntt == 2)] = 1
testData.ntt[(testData.ntt == 1) | (testData.ntt == 2)] =1

trainData.ntt[trainData.ntt == 3] = 2
testData.ntt[testData.ntt == 3] = 2

trainData.ntt[(trainData.ntt >= 4) & (trainData.ntt <= 6)] = 3
testData.ntt[(testData.ntt >= 4) & (testData.ntt <= 6)] = 3

#orientation
trainData.orientation[(trainData.orientation == 90) |(trainData.orientation == 2)] = 0
testData.orientation[(testData.orientation == 90) | (testData.orientation == 2)] = 0

"""
组合特征
"""
"""
交叉特征
"""
label = trainData.pop("label")
train_sid = trainData.pop("sid")
test_sid = testData.pop("sid")

#删除部分特征
trainData = trainData.drop(["os"],axis =1)
testData = testData.drop(["os"],axis =1)

allData = pd.concat([trainData, testData],axis= 0)

#TODO:2019.8.10号添加
allData["req_ip"] = allData.groupby("reqrealip")["ip"].transform("count")
allData['model'].replace('PACM00', "OPPO", inplace=True)
allData['model'].replace('PBAM00', "OPPO", inplace=True)
allData['model'].replace('PBEM00', "OPPO", inplace=True)
allData['model'].replace('PADM00', "OPPO", inplace=True)
allData['model'].replace('PBBM00', "OPPO", inplace=True)
allData['model'].replace('PAAM00', "OPPO", inplace=True)
allData['model'].replace('PACT00', "OPPO", inplace=True)
allData['model'].replace('PABT00', "OPPO", inplace=True)
allData['model'].replace('PBCM10', "OPPO", inplace=True)

for feat in ["model","make","lan"]:
    allData[feat] = allData[feat].astype(str)
    allData[feat] = allData[feat].map(lambda x:x.upper())
allData["big_model"] = allData["model"].map(lambda x:x.split(' ')[0])

feature = "adunitshowid"
allData[feature+"_0"], allData[feature+"_1"], allData[feature+"_2"], allData[feature+"_3"] = allData[feature].apply(lambda x: x[0:8]),\
                                                                                             allData[feature].apply(lambda x: x[8:16]),\
                                                                                             allData[feature].apply(lambda x: x[16:24]),\
                                                                                             allData[feature].apply(lambda x: x[24:32])
feature = "pkgname"
allData[feature+"_1"], allData[feature+"_2"], allData[feature+"_3"] = allData[feature].apply(lambda x: x[8:16]),\
                                                                       allData[feature].apply(lambda x: x[16:24]),\
                                                                       allData[feature].apply(lambda x: x[24:32])
feature = "mediashowid"
allData[feature+"_0"], allData[feature+"_1"], allData[feature+"_2"], allData[feature+"_3"] = allData[feature].apply(lambda x: x[0:8]),\
                                                                                             allData[feature].apply(lambda x: x[8:16]),\
                                                                                             allData[feature].apply(lambda x: x[16:24]),\
                                                                                             allData[feature].apply(lambda x: x[24:32])
feature = "idfamd5"
allData[feature+"_1"], allData[feature+"_2"], allData[feature+"_3"] = allData[feature].apply(lambda x: x[8:16]),\
                                                                      allData[feature].apply(lambda x: x[16:24]),\
                                                                      allData[feature].apply(lambda x: x[24:32])

feature = "macmd5"
allData[feature + "_0"], allData[feature + "_1"], allData[feature + "_3"] = allData[feature].apply(lambda x: x[0: 8]), allData[feature].apply(lambda x: x[8: 16]), \
                                                             allData[feature].apply(lambda x: x[24:32])


#对上述特征进行类别编码
for fe in ["big_model","adunitshowid_0","adunitshowid_1","adunitshowid_2","adunitshowid_3","pkgname_1","pkgname_2","pkgname_3","mediashowid_0","mediashowid_1","mediashowid_2","mediashowid_3","idfamd5_1","idfamd5_2","idfamd5_3",\
          "macmd5_0","macmd5_1","macmd5_3"]:
    le = LabelEncoder()

    #2019.8.11  -- 添加
    if fe not in ["idfamd5_1", "idfamd5_2", "idfamd5_3", "ver_len"]:
        allData[fe+"_rank"] = allData.groupby([fe])[fe].transform("count").rank(method="min")
    if fe not in ["idfamd5_1", "idfamd5_2", "idfamd5_3", "macmd5_1", "macmd5_2", "macmd5_3", "ver_len"]:
        allData[fe+"_count"] = allData.groupby([fe])[fe].transform("count")
    #----------------------------
    le.fit(allData[fe])
    allData[fe] = le.transform(allData[fe].astype("str"))

allData["size"] = (np.sqrt(allData["h"]**2 + allData["w"] ** 2)/ 2.54) / 1000
allData["ratio"] = allData["h"] / allData["w"]
allData["px"] = allData["ppi"] * allData["size"]
allData["area"] = allData["h"] * allData["w"]

allData["ver_len"] = allData["ver"].apply(lambda x:str(x).split(".")).apply(lambda x:len(x))
allData["osv_len"] = allData["osv"].apply(lambda x:str(x).split(".")).apply(lambda x:len(x))


#TODO:2019.8.10=---------------------------------------------------------------------------------------------------------------------------------------------------

#TODO:2019.8.11--------------特征构造代码开始-------------------------------------------------------- -------------------------------

#主要是对于新生成的特征进行rank,count,skew,cat特征构造，查看效果
#quantile
class_num = 8
quantile = []
for i in range(class_num+1):
    quantile.append(allData["ratio"].quantile(q=i / class_num))

allData["ratio_cat"] = allData["ratio"]
for i in range(class_num + 1):
    if i != class_num:
        allData["ratio_cat"][((allData["ratio"] < quantile[i + 1]) & (allData["ratio"] >= quantile[i]))] = i
    else:
        allData["ratio_cat"][((allData["ratio"] == quantile[i]))] = i - 1

allData["ratio_cat"] = allData["ratio_cat"].astype(str).fillna("0.0")
le = LabelEncoder()
le.fit(allData["ratio_cat"])
allData["ratio_cat"] = le.transform(allData["ratio_cat"])
del le

class_num = 10
quantile = []
for i in range(class_num + 1):
    quantile.append(allData["area"].quantile(q=i / class_num))

allData["area_cat"] = allData["area"]
for i in range(class_num + 1):
    if i != class_num:
        allData["area_cat"][((allData["area"] < quantile[i + 1]) & (allData["area"] >= quantile[i]))] = i
    else:
        allData["area_cat"][((allData["area"] == quantile[i]))] = i - 1

allData["area_cat"] = allData["area_cat"].astype(str).fillna("0.0")
le = LabelEncoder()
le.fit(allData["area_cat"])
allData["area_cat"] = le.transform(allData["area_cat"])
del le

class_num = 10
quantile = []
for i in range(class_num + 1):
    quantile.append(allData["size"].quantile(q=i / class_num))

allData["size_cat"] = allData["size"]
for i in range(class_num + 1):
    if i != class_num:
        allData["size_cat"][((allData["size"] < quantile[i + 1]) & (allData["size"] >= quantile[i]))] = i
    else:
        allData["size_cat"][((allData["size"] == quantile[i]))] = i - 1

allData["size_cat"] = allData["size_cat"].astype(str).fillna("0.0")
le = LabelEncoder()
le.fit(allData["size_cat"])
allData["size_cat"] = le.transform(allData["size_cat"])

del le
gc.collect()

#TODO:2019.8.11--------------特征构造代码结束----------------------------------------------------------------------------------------

#TODO:2019.8.12-----利用LGB和XGB来生成叶子节点特征
#使用apply函数进行生成，然后再与原特征进行融合。。。。

"""
聚合特征主要尝试以下：
    ip
    *id
"""
#TODO:统计特征
#nunique计算
adid_feat_nunique = ["mediashowid","apptype","city","ip","reqrealip","province","model","dvctype","make","ntt","carrier","osv","lan"]

for feat in adid_feat_nunique:
    gp1 = allData.groupby("adunitshowid")[feat].nunique().reset_index()
    gp1.columns = ["adunitshowid","adid_nuni_"+feat]
    allData = allData.merge(gp1, how = "left",on="adunitshowid")

gp2 = allData.groupby("mediashowid")["adunitshowid"].nunique().reset_index()
gp2.columns = ["mediashowid","meid_adid_nuni"]
allData = allData.merge(gp2, how = "left", on = "mediashowid")

gp2 = allData.groupby("city")["adunitshowid"].nunique().reset_index()
gp2.columns = ["city","city_adid_nuni"]
allData = allData.merge(gp2, how = "left", on = "city")

gp2 = allData.groupby("province")["adunitshowid"].nunique().reset_index()
gp2.columns = ["province","province_adid_nuni"]
allData = allData.merge(gp2, how = "left", on = "province")

gp2 = allData.groupby("ip")["adunitshowid"].nunique().reset_index()
gp2.columns = ["ip","ip_adid_nuni"]
allData = allData.merge(gp2, how = "left", on = "ip")

gp2 = allData.groupby("model")["adunitshowid"].nunique().reset_index()
gp2.columns = ["model","model_adid_nuni"]
allData = allData.merge(gp2, how = "left", on = "model")

gp2 = allData.groupby("make")["adunitshowid"].nunique().reset_index()
gp2.columns = ["make","make_adid_nuni"]
allData = allData.merge(gp2, how = "left", on = "make")


del gp1
del gp2
gc.collect()

#根据对外媒体id进行类别计数
meid_feat_nunique = ["adunitshowid","apptype","city","ip","reqrealip","province","model","dvctype","make","ntt","carrier","osv","lan"]
for feat in meid_feat_nunique:
    gp1 = allData.groupby("mediashowid")[feat].nunique().reset_index()
    gp1.columns = ["mediashowid","medi_nuni_"+feat]
    allData = allData.merge(gp1, how = "left",on="mediashowid")
gp2 = allData.groupby("city")["mediashowid"].nunique().reset_index()
gp2.columns = ["city","city_medi_nuni"]
allData = allData.merge(gp2, how = "left", on = "city")

gp2 = allData.groupby("ip")["mediashowid"].nunique().reset_index()
gp2.columns = ["ip","ip_medi_nuni"]
allData = allData.merge(gp2, how = "left", on = "ip")

gp2 = allData.groupby("province")["mediashowid"].nunique().reset_index()
gp2.columns = ["province","province_medi_nuni"]
allData = allData.merge(gp2, how = "left", on = "province")

gp2 = allData.groupby("model")["mediashowid"].nunique().reset_index()
gp2.columns = ["model","model_medi_nuni"]
allData = allData.merge(gp2, how = "left", on = "model")

gp2 = allData.groupby("make")["mediashowid"].nunique().reset_index()
gp2.columns = ["make","make_medi_nuni"]
allData = allData.merge(gp2, how = "left", on = "make")

del gp1
del gp2
gc.collect()

#adidmd5
adidmd5_feat_nunique = ["apptype","city","ip","reqrealip","province","model","dvctype","make","ntt","carrier","osv","lan"]
for feat in adidmd5_feat_nunique:
    gp1 = allData.groupby("adidmd5")[feat].nunique().reset_index()
    gp1.columns = ["adidmd5","android_nuni_"+feat]
    allData =allData.merge(gp1, how= "left", on = "adidmd5")


gp2 = allData.groupby("city")["adidmd5"].nunique().reset_index()
gp2.columns = ["city","city_adidmd_nuni"]
allData = allData.merge(gp2, how = "left", on = "city")

gp2 = allData.groupby("ip")["adidmd5"].nunique().reset_index()
gp2.columns = ["ip","ip_adidmd_nuni"]
allData = allData.merge(gp2, how = "left", on = "ip")

gp2 = allData.groupby("province")["adidmd5"].nunique().reset_index()
gp2.columns = ["province","province_adidmd_nuni"]
allData = allData.merge(gp2, how = "left", on = "province")

gp2 = allData.groupby("model")["adidmd5"].nunique().reset_index()
gp2.columns = ["model","model_adidmd_nuni"]
allData = allData.merge(gp2, how = "left", on = "model")

gp2 = allData.groupby("make")["adidmd5"].nunique().reset_index()
gp2.columns = ["make","make_adidmd_nuni"]
allData = allData.merge(gp2, how = "left", on = "make")

del gp1
del gp2
gc.collect()

#TODO:进行每个特征中不同取值的数量  ----2019.8.5
city_cnt = allData["city"].value_counts().to_dict()
allData["city_cnt"] = allData["city"].map(city_cnt)

model_cnt = allData["model"].value_counts().to_dict()
allData["model_cnt"] = allData["model"].map(model_cnt)

make_cnt = allData["make"].value_counts().to_dict()
allData["make_cnt"] = allData["make"].map(make_cnt)

ip_cnt = allData["ip"].value_counts().to_dict()
allData["ip_cnt"] = allData["ip"].map(ip_cnt)

reqrealip_cnt = allData["reqrealip"].value_counts().to_dict()
allData["reqrealip_cnt"] = allData["reqrealip"].map(reqrealip_cnt)

osv_cnt = allData["osv"].value_counts().to_dict()
allData["osv_cnt"] = allData["osv"].map(osv_cnt)

#TODO:ratio特征构造

#TODO:关于mean特征的构造
for fe in []:
    pass





#TODO:关于std特征的构造

#TODO:交叉特征
feat_1 = ["adunitshowid","mediashowid","adidmd5"]
feat_2 = ["apptype","city","ip","reqrealip","province","model","dvctype","make","ntt","carrier","osv","lan"]
cross_feat = []
for fe_1 in feat_1:
    for fe_2 in feat_2:
        col_name = "cross_"+fe_1+"_and_"+fe_2
        cross_feat.append(col_name)
        allData[col_name] = allData[fe_1].astype(str).values + "_" + allData[fe_2].astype(str).values

#TODO:对交叉特征进行计数  ---  2019.8.7
for fe in cross_feat:
    locals()[fe+"_cnt"] = allData[fe].value_counts().to_dict()
    allData[fe+"_cnt"] = allData[fe].map(locals()[fe+"_cnt"])


for fe in cross_feat:
    le_feat = LabelEncoder()
    le_feat.fit(allData[fe])
    allData[fe] = le_feat.transform(allData[fe])

#先对连续特征进行归一化等操作
continue_feats = ["h","w","ppi","nginxtime","cost_time"]
#需要进行类别编码的特征
label_encoder_feats = ["ver","lan","pkgname","adunitshowid","mediashowid","apptype","city","province","ip","reqrealip","adidmd5","imeimd5","idfamd5","openudidmd5","macmd5","model","make","osv"]
#需要进行独热编码的特征
cate_feats = ["dvctype","ntt","carrier"]

#TODO:2019.8.8添加
for cf in list(set(cate_feats+["ver","lan","adunitshowid","mediashowid","apptype","adidmd5","imeimd5","idfamd5","openudidmd5","macmd5"])):
    locals()[cf+"_cnt"] = allData[cf].value_counts().to_dict()
    allData[cf+"_cnt"] = allData[cf].map(locals()[cf+"_cnt"])

#labelencoder转化
for fe in ["pkgname","adunitshowid","mediashowid","apptype","province","ip","reqrealip","adidmd5","imeimd5","idfamd5","openudidmd5","macmd5","ver","lan"]:
    le_feat = LabelEncoder()
    le_feat.fit(allData[fe])
    allData[fe] = le_feat.transform(allData[fe])

#由于city在类别编码时出现错误，所以我们就手动进行编码
city_2_idx = dict(zip(list(set(allData["city"].unique())),range(len(list(set(allData["city"].unique()))))))

#对model, make,osv 自定义编码方式，因为用类别编码的时候，出现了报错，主要是由于取值中含有不可识别字符，后期再进行处理一下
model_2_idx = dict(zip(list(set(allData["model"].unique())),range(len(list(set(allData["model"].unique()))))))
make_2_idx = dict(zip(list(set(allData["make"].unique())),range(len(list(set(allData["make"].unique()))))))
osv_2_idx = dict(zip(list(set(allData["osv"].unique())),range(len(list(set(allData["osv"].unique()))))))

allData["city"] = allData["city"].map(city_2_idx)
allData["model"] = allData["model"].map(model_2_idx)
allData["make"] = allData["make"].map(make_2_idx)
allData["osv"] = allData["osv"].map(osv_2_idx)

# #对连续变量进行简单的归一化处理
for fe in continue_feats:
    temp_data = np.reshape(allData[fe].values,[-1,1])
    mm = MinMaxScaler()
    mm.fit(temp_data)
    allData[fe] = mm.transform(temp_data)

#对运营商，网络类型进行简单预处理
allData['carrier'].value_counts()
allData["carrier"] = allData["carrier"].map({0.0:0,-1.0:0,46000.0:1,46001.0:2,46003.0:3}).astype(int)
allData["ntt"] = allData["ntt"].astype(int)
allData["orientation"] = allData["orientation"].astype(int)
allData["dvctype"] = allData["dvctype"].astype(int)


#删除特征较低的部分特征
allData.pop("idfamd5")
allData.pop("adid_nuni_mediashowid")
allData.pop("adid_nuni_apptype")
allData.pop("medi_nuni_apptype")

allData.pop('medi_nuni_adunitshowid')
allData.pop('province_adidmd_nuni')

allData.pop("android_nuni_osv")


#2019.8.3添加删除的列 ---- 测试,貌似有点涨分
allData.pop('province_medi_nuni')
allData.pop('android_nuni_province')
allData.pop('android_nuni_dvctype')
allData.pop('android_nuni_carrier')

#将处理后的数据保存起来
print("saving the file.....")

allData.to_csv("Data/allData.csv",index=False)
end = time.time()
print("saving finished. it costs {}".format(datetime.timedelta(seconds=end-start)))