'''
@author: j.langlois
'''

import numpy as np
import theano
import theano.tensor as T
import lasagne
import sys
import batch_functions
from pandas.util.testing import network
import euler 
import createLayers
import os
import threading

sys.setrecursionlimit(50000)
      
class Network():
    
    def __init__(self,dimChannels,dimFeatures,dimOutput,name="default_network",networkFile='networks/network1.txt'):
        self.name=name
        self.dimChannels=dimChannels
        self.dimFeatures=dimFeatures
        self.dimOutput=dimOutput
        self._observers = []

        input_var = T.tensor4('inputs')
        target_var = T.matrix('targets')
        learning_rate=T.scalar('learning rate')
        regularization_weight=T.scalar('learning rate')
                
        l_input = lasagne.layers.InputLayer(shape=(None, dimChannels, dimFeatures, dimFeatures),input_var=input_var)
        core=createLayers.createLayers(networkFile,l_input)
        l_output = lasagne.layers.DenseLayer(core,num_units=dimOutput,nonlinearity=lasagne.nonlinearities.tanh,W=lasagne.init.Orthogonal())
            
        print("CREATING FUNCTIONS...")
        prediction = lasagne.layers.get_output(l_output)
        cost = lasagne.objectives.squared_error(prediction,target_var)
        l2_penalty = lasagne.regularization.regularize_network_params(l_output, lasagne.regularization.l2)
        loss = cost.mean()+regularization_weight*l2_penalty
        
        params = lasagne.layers.get_all_params(l_output, trainable=True)
        updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=learning_rate, momentum=0.9)

        test_prediction = lasagne.layers.get_output(l_output, deterministic=True)
        test_cost = lasagne.objectives.squared_error(test_prediction,target_var)
        test_loss=test_cost.mean()
        
        self.last_layer=l_output
        self.f_train=theano.function([input_var, target_var,learning_rate,regularization_weight], loss, updates=updates)
        self.f_predict=theano.function([input_var], test_prediction)
        self.f_accuracy=theano.function([input_var,target_var],test_loss)

        print("FUNCTIONS CREATED.")
        
    def createTrainingThread(self,train_data,train_labels,valid_data,valid_labels,trigger,
                             batchSize=30,epochs=500,learningRate=0.01,penalty=0.000001):
        self.trainingThread = threading.Thread(None, self.trainNetwork, None,(),
                                               {'train_data':train_data,
                                                'train_labels':train_labels,
                                                'valid_data':valid_data,
                                                'valid_labels':valid_labels,
                                                'trigger':trigger,
                                                'batchSize':batchSize,
                                                'epochs':epochs,
                                                'learningRate':learningRate,
                                                'penalty':penalty})
    
    def trainNetwork(self,train_data,train_labels,valid_data,valid_labels,trigger,batchSize=20,epochs=100,learningRate=0.001,penalty=0.001):
        print("TRAINING STARTS")
        
        epoch_kept=0
        min_valid_error=10000
        valid_data=batch_functions.normalize_batch(valid_data)
        
        for e in range(0,epochs):
            f=open(self.name+'_data.txt','a')
            train_err = 0
            train_batches = 0
            valid_error=0
            for batch in batch_functions.iterate_minibatches(train_data, train_labels, batchSize, shuffle=True):
                inputs, targets = batch
                inputs=batch_functions.normalize_batch(inputs)
                train_err += self.f_train(inputs, targets, learningRate,penalty)
                train_batches += 1
            valid_error=self.f_accuracy(valid_data,valid_labels)
            learningRate=learningRate/1.005
            
            if valid_error<min_valid_error:
                min_valid_error=valid_error
                epoch_kept=e

            f.write(str(e)+';'+str(train_err)+';'+str(valid_error)+'\n')
            f.close()
            self._observers[0]()
            print("EPOCH : "+str(e)+'    LOSS: '+str(train_err)+'    ERROR: '+str(valid_error)+'    RATE: '+str(learningRate))
        print('END - EPOCH KEPT : '+str(epoch_kept))   
        
        return 1
    
    def predict(self,test_data):
        batch_functions.normalize_batch(test_data)
        return self.f_predict(test_data)
    
    def testNetwork(self,test_data,test_labels):
        test_data=batch_functions.normalize_batch(test_data)
        prediction=self.f_predict(test_data)
        #print(test_labels,prediction)
        
        moy=0
        for i in range(prediction.shape[0]):
            q=[test_labels[i,0],test_labels[i,1],test_labels[i,2],test_labels[i,3]]
            anglesR=np.array(euler.quat2euler(prediction[i,:]))
            anglesV=np.array(euler.quat2euler(q))
            for j in range(3):
                anglesR[j]=anglesR[j]*180/np.pi
                if anglesR[j]<0:
                    anglesR[j]=anglesR[j]+360.0
                anglesV[j]=anglesV[j]*180/np.pi
                if anglesV[j]<0:
                    anglesV[j]=anglesV[j]+360.0
            #print(anglesR,anglesV)
            anglesD=[]
            for j in range(3):
                angle = np.abs(anglesR[j] - anglesV[j]) % 360;  
                if angle > 180 :
                    anglesD.append(360-angle)
                else: 
                    anglesD.append(angle)
            #print(anglesD)
            moy+=np.mean(anglesD)
        moy/=prediction.shape[0]
        return moy
                
