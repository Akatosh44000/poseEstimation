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
import sys

max_network=int(sys.argv[1])
iterations=int(sys.argv[2])

home = expanduser("~")
MODEL_PATH=home+'/DATASETS'
DATASET='MULTITUDE'

MODEL_OBJECT='BREATHER'
DIMENSION='SAMPLES'
DEPTH=False

[train_data,train_labels]=datasetImportConv(MODEL_PATH,DATASET,MODEL_OBJECT,DIMENSION,'TRAIN',DEPTH)
[valid_data,valid_labels]=datasetImportConv(MODEL_PATH,DATASET,MODEL_OBJECT,DIMENSION,'VALIDATION',DEPTH)
[test_data,test_labels]=datasetImportConv(MODEL_PATH,DATASET,MODEL_OBJECT,DIMENSION,'TEST',DEPTH)

iterations=3
for id in range(1,max_network+1):
    total_error=0
    name='network'+str(id)
    networkFile='networks/network'+str(id)+'.txt'
    if os.path.isfile(name+"_data.txt"):
        os.remove(name+"_data.txt")
    for _ in range(iterations):
        network1=Network(train_data.shape[1],train_data.shape[2],train_labels.shape[1],name,networkFile=networkFile)
        network1.trainNetwork(train_data,train_labels,valid_data,valid_labels,batchSize=100,epochs=150,learningRate=0.01,penalty=0.01)
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

