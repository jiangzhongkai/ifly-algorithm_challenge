# ifly-algorithm_challenge
讯飞移动反欺诈算法竞赛,目前分数只有94.41

讯飞移动反欺诈算法数据竞赛网址： http://challenge.xfyun.cn/2019/gamedetail?type=detail/mobileAD

### 总体流程
```
  | EDA
  | 数据预处理
  | 数据特征构造
  | 模型搭建
  | 模型参数的调优以及特征筛选
```
#### EDA

 在做数据竞赛的时候，当我们拿到数据集的时候，我们首先要做的，也是最重要的事情那就是进行数据探索分析，这部分做的好，对接下来的每个部分的工作都是有力的参考，而不会盲目的去进行数据预处理，以及特征构造。数据探索分析主要做以下几个部分的工作:
   - 查看每个数据特征的缺失情况、特征中是否含有异常点，错误值
   - 查看每个特征的分布情况，主要包括连续特征是否存在偏移，也就是不服从正态分布的情况；离散特征的值是否具体分布如何
   - 查看一般特征之间的相关性如何
   - 查看一般特征与目标特征之间的相关性


#### 数据预处理

通过EDA,我们对数据进行了初步分析，接下来就针对EDA部分得出的结果，来进行数据的预处理工作,主要做了以下工作:
   - 


### 参考资料

- [x] GBDT、xgboost对比分析：

https://wenku.baidu.com/view/f3da60b4951ea76e58fafab069dc5022aaea463e.html

- [x] xgboost论文：

https://arxiv.org/pdf/1603.02754.pdf

- [x] lightgbm论文:

http://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf

- [x] catboost论文：

  - https://arxiv.org/pdf/1706.09516.pdf

  - http://learningsys.org/nips17/assets/papers/paper_11.pdf
