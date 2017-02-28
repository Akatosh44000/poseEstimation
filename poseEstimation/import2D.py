# -*- coding: utf-8 -*-
'''
@author: Julien LANGLOIS
IRCCyN
'''

import numpy as np
import cv2
import os
from random import shuffle
import pickle

def importData(pathTo):
    dictionnary=pickle.load(open(pathTo+'.data', "rb"))
    return dictionnary

def prepareData(data):
    mean=np.mean(data['data'])
    out=np.empty([len(data['data']),data['data'][0].shape[1]],dtype=np.float32)
    for i,d in enumerate(data['data']):
        img=np.array(d)
        size=np.int(np.sqrt(img.shape[1]))
        img=img.reshape((size,size))
        h, w = img.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        img=cv2.floodFill(img,mask,(0,0),0)[1]
        img=cv2.floodFill(img,mask,(w-1,h-1),0)[1]
        img=cv2.floodFill(img,mask,(w-1,1),0)[1]
        img=cv2.floodFill(img,mask,(0,h-1),0)[1]
        img=cv2.adaptiveThreshold(img,255,1,1,11,2)
        img=np.reshape(img,(size*size))
        img=img-mean
        out[i,:]=img/float(np.max(np.abs(img)))
    return out

def prepareDataM(data):
    mean=np.mean(data['data'])
    out=np.empty([len(data['data']),data['data'][0].shape[1]],dtype=np.float32)
    for i,d in enumerate(data['data']):
        mat=np.array(d)
        mat=mat-mean
        out[i,:]=mat/float(np.max(np.abs(mat)))
    return out

def datasetImport(dataset_path,dataset,dataset_object,dimension,typeOfSet):
    pathTo=dataset_path+'/'+dataset+'/'+dataset_object+'/'+dimension+'/'+typeOfSet
    data=importData(pathTo)
    label=data['label']
    data=prepareData(data)
    return [np.matrix(data),np.matrix(label)]
    
def singleShotImport(dataset_path,dataset,dataset_object,name):
    pathTo=dataset_path+'\\'+dataset+'\\'+dataset_object+'\\'+name
    data=cv2.imread(pathTo+'.png',0)
    data=cv2.resize(data,(64,64))
    data=np.reshape(data,data.shape[0]*data.shape[1])
    data=data-np.mean(data)
    data=data/float(np.max(np.abs(data)))
    label=[np.cos(21*np.pi/180.0),np.sin(21*np.pi/180.0),np.cos(147*np.pi/180.0),np.sin(147*np.pi/180.0)]
    return [np.matrix(data),np.matrix(label)]

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
    
    
    
    