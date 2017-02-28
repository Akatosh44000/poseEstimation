# -*- coding: utf-8 -*-
'''
@author: Julien LANGLOIS
IRCCyN
'''

from network_conv_estimation import Network
from import2D import datasetImportConv
import pickle
import batch_functions
import os
import shapeData
from os.path import expanduser

home = expanduser("~")
MODEL_PATH=home+'/DATASETS'
DATASET='MULTITUDE'

MODEL_OBJECT='BREATHER'
DIMENSION='SAMPLES'
DEPTH=False
TEST=False

[train_data,train_labels]=datasetImportConv(MODEL_PATH,DATASET,MODEL_OBJECT,DIMENSION,'TRAIN',DEPTH)
[valid_data,valid_labels]=datasetImportConv(MODEL_PATH,DATASET,MODEL_OBJECT,DIMENSION,'VALIDATION',DEPTH)
[test_data,test_labels]=datasetImportConv(MODEL_PATH,DATASET,MODEL_OBJECT,DIMENSION,'TEST',DEPTH)
#[test_dataV,test_labelsV]=datasetImportConv(MODEL_PATH,DATASET,MODEL_OBJECT,'VRAC','TEST',DEPTH)

if TEST:
    f=open('network1.dat','rb')
    networkF=pickle.load(f)
    test_data=batch_functions.normalize_batch(test_data)
    res=networkF.testNetwork(test_data,test_labels)
    print(res)

else:
    iterations=3
    max_network=2
    for id in range(max_network):
        total_error=0
        name='network'+str(id)
        if os.path.isfile(name+"_data.txt"):
            os.remove(name+"_data.txt")
        for _ in range(iterations):
            network1=Network(train_data.shape[1],train_data.shape[2],train_labels.shape[1],name,networkFile='networks/network'+str(id)+'.txt')
            network1.trainNetwork(train_data,train_labels,valid_data,valid_labels,batchSize=10,epochs=100,learningRate=0.01,penalty=0.01)
            error=network1.testNetwork(test_data,test_labels)
            f=open(name+'_data.txt','a')
            f.write(str(error)+'\n')
            f.close()
            print("CURRENT TEST ERROR : " + str(error))
            total_error+=error
            del network1
        shapeData.formatData(name+"_data.txt")
        total_error/=iterations
        print(total_error)

