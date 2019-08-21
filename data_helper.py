"""-*- coding: utf-8 -*-
 DateTime   : 2019/7/28 16:03
 Author  : Peter_Bonnie
 FileName    : data_helper.py
 Software: PyCharm
"""
#数据处理文件
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import time
import re


PATH = "Data/"

#TODO:缺失值填充

def my_fillna(df):
    for i in df.columns:
        if df[i].isnull().sum() / df[i].values.shape[0] > 0.0:
            if df[i].dtype == "object":
                df[i] = df[i].fillna(str(-1))
                df[i] = df[i].replace("nan", str(-1))
            elif df[i].dtype == "float":
                df[i] = df[i].fillna(-1.0)
                df[i] = df[i].replace("nan", -1.0)
            else:
                df[i] = df[i].fillna(-1)
                df[i] = df[i].replace("nan", -1)
    return df

#TODO:操作系统版本号进行更细致的分解
def split_osv_into_threeParts(x):

    x = list(map(int, x.split('.')))
    while len(x) < 3:
        x.append(0)

    return x[0], x[1], x[2]


def remove_lowcase(se):
    count = dict(se.value_counts())
    se = se.map(lambda x: -1 if count[x] < 5 else x)
    return se


def clearn_make(make):
    if 'oppo' in make:
        return 'oppo'
    elif 'vivo' in make:
        return 'vivo'
    elif 'huawei' in make or 'honor' in make:
        return 'huawei'
    elif 'redmi' in make:
        return 'xiaomi'

    strs = make.split()
    if len(strs) > 1:
        s = strs[0]
        if s == 'mi' or s == 'm1' or s == 'm2' or s == 'm3' or s == 'm6':
            s = 'xiaomi'
        return s
    return make


def clearn_model(model):
    if '%' in model:
        return model.split('%')[0]
    elif 'vivo' in model:
        return 'vivo'
    elif 'oppo' in model or 'pb' in model or 'pa' in model or 'pc' in model:
        return 'oppo'
    elif 'huawei' in model or 'honor' in model:
        return 'huawei'
    elif 'redmi' in model or 'xiaomi' in model or 'm5' in model or 'm4' in model or 'm7' in model or 'mi' in model or 'm2' in model or 'm3' in model or 'm6' in model:
        return 'xiaomi'
    elif 'letv' in model:
        return 'letv'
    elif 'oneplus' in model:
        return 'oneplus'
    elif 'zte' in model or '中兴' in model:
        return 'zte'
    # 使用正则来匹配
    model = re.compile(r'v.*.[at]').sub('vivo', model)

    return model


def padding_empty_osv(osv):
    str_osv = osv.strip().split('.')
    while len(str_osv) < 3:
        str_osv.append('0')
    if str_osv[0] == '':
        str_osv[0] = '0'
    if str_osv[1] == '':
        str_osv[1] = '0'
    return '.'.join(str_osv)


def remove_outlier_osv(osv):
    if "吴氏家族版4.4.2" in osv:
        osv = "4.4.2"
    elif '3.2.0-2-20180726.9015' in osv:
        osv = "3.2.0"
    elif '21100.0.0' in osv or '21000.0.0' in osv:
        osv = "0.0.0"
    elif len(osv.split('.')) > 3:
        osv = '.'.join(osv.split('.')[:3])
    return osv


#计算运行时间
def compute_cost(sec):
    hours,secs = divmod(sec, 3600)
    mins,secs = divmod(secs, 60)
    return "Fininshed, and it cost {0} hours : {1} mins : {2} secs".format(int(hours),int(mins),int(secs))


def making(x):
    x = x.lower()
    if 'iphone' in x or 'apple' in x or '苹果' in x:
        return 'iphone'
    elif 'huawei' in x or 'honor' in x or '华为' in x or '荣耀' in x:
        return 'huawei'
    elif 'xiaomi' in x or '小米' in x or 'redmi' in x:
        return 'xiaomi'
    elif '魅族' in x:
        return 'meizu'
    elif '金立' in x:
        return 'gionee'
    elif '三星' in x or 'samsung' in x:
        return 'samsung'
    elif 'vivo' in x:
        return 'vivo'
    elif 'oppo' in x:
        return 'oppo'
    elif 'lenovo' in x or '联想' in x:
        return 'lenovo'
    elif 'nubia' in x:
        return 'nubia'
    elif 'oneplus' in x or '一加' in x:
        return 'oneplus'
    elif 'smartisan' in x or '锤子' in x:
        return 'smartisan'
    elif '360' in x or '360手机' in x:
        return '360'
    elif 'zte' in x or '中兴' in x:
        return 'zte'
    else:
        return 'others'


# 处理'lan’
def lan(x):
    x = x.lower()
    if x in ['zh-cn','zh','cn','zh_cn','zh_cn_#hans','zh-']:
        return 'zh-cn'
    elif x in ['tw','zh-tw','zh_tw']:
        return 'zh-tw'
    elif 'en' in x:
        return 'en'
    elif 'hk' in x:
        return 'zh-hk'
    else:
        return x
























