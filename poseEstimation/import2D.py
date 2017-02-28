# -*- coding: utf-8 -*-
'''
@author: Julien LANGLOIS
IRCCyN
'''

import numpy as np
import pickle

def importData(pathTo):
    dictionnary=pickle.load(open(pathTo+'.data', "rb"))
    return dictionnary

def datasetImportConv(dataset_path,dataset,dataset_object,dimension,typeOfSet,depth=False):

    pathTo=dataset_path+'/'+dataset+'/'+dataset_object+'/'+dimension+'/'+typeOfSet
    dict=importData(pathTo)
    [data,labels]=[dict['data'],dict['labels']]
    if depth==True:
        dataTensor=np.zeros([len(data),2,data[0].shape[1],data[0].shape[2]],np.float32)
    else:
        dataTensor=np.zeros([len(data),1,data[0].shape[1],data[0].shape[2]],np.float32)
    for i,d in enumerate(data):
        if depth==True:
            dataTensor[i,0,:,:]=np.matrix(d[0,:,:],np.float32)
            dataTensor[i,1,:,:]=np.matrix(d[1,:,:],np.float32)
        else:
            dataTensor[i,0,:,:]=np.matrix(d[0,:,:],np.float32)
    return [dataTensor,np.matrix(labels,np.float32)]
    
    
    
    