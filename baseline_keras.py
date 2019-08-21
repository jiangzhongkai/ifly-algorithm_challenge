"""-*- coding: utf-8 -*-
 DateTime   : 2019/8/20 9:17
 Author  : Peter_Bonnie
 FileName    : baseline_keras.py
 Software: PyCharm
"""
from keras.models import  Sequential, Model
from keras.layers import  Dense, Dropout, Activation, BatchNormalization, Input
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
import gc
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold,StratifiedKFold
import  datetime
import keras
from keras import backend as K

#load data
all_data = pd.read_csv("Data/allData.csv")

train_data = pd.read_csv("Data/train_data.txt",delimiter='\t')
test_data = pd.read_csv("Data/test_data.txt",delimiter='\t')

trainData = all_data[:train_data.shape[0]]
testData = all_data[train_data.shape[0]:]
test_sid = test_data["sid"]

label = train_data["label"]
del train_data["label"]

#model
K.clear_session()

class MLP(object):

    def __init__(self,drop_out,activation,input_units,trainX, testX,trainY,epoch,batchsize,num_class,optimizer,loss,metrics,valX,valY):

        self.drop_out = drop_out
        self.activation = activation
        self.input_units = input_units
        self.trainX = trainX
        self.trainY = trainY
        self.testX = testX
        self.valX = valX
        self.valY = valY
        self.epoch = epoch
        self.batchsize = batchsize
        self.num_class = num_class
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

    def  _build(self):
        model = Sequential()
        #first layer
        model.add(Dense(self.input_units,input_dim=self.trainX.shape[1],activation=self.activation))
        model.add(BatchNormalization())
        model.add(Dropout(self.drop_out))
        #second layer
        model.add(Dense(self.input_units // 2, activation= self.activation))
        model.add(Dropout(self.drop_out))

        #third layer
        model.add(Dense(self.input_units // 4, activation= self.activation))
        model.add(Dropout(self.drop_out))

        #forth layer
        model.add(Dense(self.input_units // 8, activation= self.activation))
        model.add(Dropout(self.drop_out / 2))

        #output
        model.add(Dense(self.num_class,activation="sigmoid"))

        model.compile(optimizer=self.optimizer,loss= self.loss,metrics=self.metrics)
        model.summary()

        return model

    def _fit(self):

        model = self._build()
        history = model.fit(self.trainX, self.trainY,batch_size=self.batchsize,epochs=self.epoch,
                            callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss",min_delta=0,patience=100,verbose=1,mode="auto"),]
                            ,shuffle= True,verbose=1)
        return model

    def _predict(self):

        val_score = []
        model = self._fit()
        valPredict = model.predict(self.valX,batch_size=self.batchsize)
        val_score = f1_score(self.valY, valPredict)
        print(val_score)
        test_predict = model.predict(self.testX,batch_size=self.batchsize)
        return test_predict


skf = StratifiedKFold(n_splits=7,shuffle=True,random_state=2019)

trainX = trainData.values
testX = testData.values
trainY = label

test_pred = []
for idx, (trx,valx) in enumerate(skf.split(trainX,trainY)):
    print("=================fold_{}=====================================".format(str(idx+1)))
    X_train = trainX[trx]
    Y_train = trainY[trx]

    X_val = trainX[valx]
    Y_val = trainY[valx]

    mlp = MLP(drop_out=0.2,activation="relu",input_units=256,trainX=X_train,testX=testX,trainY=Y_train,epoch=1000,
              batchsize=128,num_class=1,optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"],valX=X_val,valY=Y_val)
    test_predict = mlp._predict()
    test_pred.append(test_predict)


submit = []
for line in np.array(test_pred).transpose():
    submit.append(np.argmax(np.bincount(line)))
final_result = pd.DataFrame(columns=["sid","label"])
final_result["sid"] = list(test_sid.unique())
final_result["label"] = submit
final_result.to_csv("submitMLP{0}.csv".format(datetime.datetime.now().strftime("%Y%m%d%H%M")),index = False)
print(final_result.head())













