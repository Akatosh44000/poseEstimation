'''
@author: j.langlois
'''
import numpy as np
import threading
import time
import requests
import http.server
import web
import os
from socketIO_client import SocketIO,BaseNamespace
import network_conv_estimation
import import2D
import scipy.signal

class networkServer:
    
    def __init__(self):
        print('WAIT:: STARTING NETWORK CLIENT')

        #BYPASS PROXY RULES FOR LOCALHOST DOMAIN
        os.environ['NO_PROXY'] = 'localhost'
        
        #NETWORK PARAMS
        self.id=0
        self.name='reseau_1'
        '''
        NETWORK CREATION...
        '''
        
        self.buildNetwork()
        self.importDataset()
        #HANDLE MESSAGE FROM SERVER
        def handle_message_from_server(args):
            message=args[0]
            self.handle_message_from_server(message)
        #HANDLE MESSAGE FROM CLIENT
        def handle_message_from_client(args):
            message=args[0]
            self.handle_message_from_client(message)
        #HANDLE REQUEST FROM CLIENT
        def handle_request_from_client(args):
            request=args[0]
            self.handle_request_from_client(request)
            
        class MessageNamespace(BaseNamespace):
            def on_connect(self):
                print('[Connected]')
            def on_disconnect(self):
                print('[Disconnected]')
            def on_MESSAGE_FROM_SERVER_TO_NETWORK(self,*args):
                handle_message_from_server(args)
            def on_MESSAGE_FROM_CLIENT_TO_NETWORK(self,*args):
                handle_message_from_client(args)
            def on_REQUEST_FROM_CLIENT_TO_NETWORK(self,*args):
                handle_request_from_client(args)

        self.socketio=SocketIO('localhost', 8080,MessageNamespace)
        print('SUCCESS:: NETWORK CLIENT STARTED')
        
        self.sendRequestToServer('getNewId',{'network_id': self.id,'network_name': self.name})
        
        
        
        
        self.socketio.wait()
    def sendMessageToserver(self,name,data):
        self.socketio.emit('MESSAGE_FROM_NETWORK_TO_SERVER',
                           {'name':name,'data':data})
    def sendRequestToServer(self,name,params):
        self.socketio.emit('REQUEST_FROM_NETWORK_TO_SERVER',
                           {'name':name,'params':params})
    def sendMessageToClient(self,name,data,client_socket_id):
        self.socketio.emit('MESSAGE_FROM_NETWORK_TO_CLIENT',
                           {'client_socket_id':client_socket_id,'name':name,'data':data})        
    def buildNetwork(self):
        print('WAIT:: BUILDING NEURAL NETWORK...')
        self.network=network_conv_estimation.Network(1,64,4)
        print('SUCCESS:: NETWORK BUILT !')
        
    def handle_message_from_server(self,message):   
        if self.id==0:
            print('INFO:: NETWORK GOT THE ID '+message['network_id']+' FROM THE SERVER.')
            self.id=int(message['network_id']) 
            
    def handle_message_from_client(self,request):
        return 1
    
    def handle_request_from_client(self,request):
        print('INFO:: HANDLING REQUEST FROM CLIENT ',request['name'])
        if request['name']=='setTrain':
            self.trainNetwork()
        if request['name']=='stopTrain':
            self.stopTraining()
        if request['name']=='getTestError':
            self.sendMessageToClient('testError',{'testError':self.testNetwork()},request['client_socket_id']);
        if request['name']=='getNetworkArchitecture':
            self.sendMessageToClient('architecture',{'architecture':self.getNetworkArchitecture()},request['client_socket_id']);
        if request['name']=='getParams':
            self.sendMessageToClient('params',{'params':self.getFakeParams(request['params'])},request['client_socket_id']);
        if request['name']=='getPipeline':
            self.sendMessageToClient('pipeline',{'pipeline':self.getPipeline(request['params'])},request['client_socket_id']);

    def getNetwork(self):    
        return self.network
    
    def importDataset(self):
        print('WAIT:: IMPORTING DATASETS...')
        MODEL_PATH='/home/akatosh/DATASETS'
        DATASET='MULTITUDE'
        MODEL_OBJECT='BREATHER'
        DIMENSION='SAMPLES'
        DEPTH=False
        self.TRAIN_DATASET=import2D.datasetImportConv(MODEL_PATH,DATASET,MODEL_OBJECT,DIMENSION,'TRAIN',DEPTH)
        self.VALIDATION_DATASET=import2D.datasetImportConv(MODEL_PATH,DATASET,MODEL_OBJECT,DIMENSION,'VALIDATION',DEPTH)
        self.TEST_DATASET=import2D.datasetImportConv(MODEL_PATH,DATASET,MODEL_OBJECT,DIMENSION,'TEST',DEPTH)
        print('SUCCESS:: DATASETS IMPORTED !')
        
    def trainNetwork(self):
        print('WAIT:: CREATING GPU TRAINING THREAD...')
        self.network.bind_to(self.newEpoch)
        trigger = threading.Event()
        self.network.createTrainingThread(self.TRAIN_DATASET[0],self.TRAIN_DATASET[1],
                                          self.VALIDATION_DATASET[0],self.VALIDATION_DATASET[1],
                                          trigger)
        print('WAIT:: STARTING GPU TRAINING THREAD...')
        self.network.trainingThread.start()
        print('SUCCESS:: TRAINING STARTED !')
        
    def stopTraining(self):
        print('WAIT:: STOPPING GPU TRAINING THREAD...')
        self.network.trainingThread._stop()
        print('SUCCESS:: TRAINING STOPPED !')
    
    def testNetwork(self):
        print('WAIT:: TESTING NETWORK...')
        error=self.network.testNetwork(self.TEST_DATASET[0], self.TEST_DATASET[1])
        print('SUCCESS:: NETWORK TESTED !')
        return error
        
    def getLoss(self):
        return self.network.loss
    
    def newEpoch(self,loss):
        print("NEW EPOCH !!!",str(loss))
        self.sendMessageToserver('newEpoch','')
        
    def getNetworkArchitecture(self):
        params=np.asarray(self.network.getParamsValues())
        architecture=[]
        for i in range(0,params.shape[0],2):
            if len(params[i].shape)>2:
                #CONVOLUTION LAYER
                architecture.append(['CONV',params[i].shape])
            else:
                architecture.append(['FC',params[i].shape])
        return architecture
    
    def getFakeParams(self,parameters):
        if len(parameters)>0:
            paramsList=[]
            params=parameters['params']
            layer=parameters['layer']
            paramsFromNetwork=np.asarray(self.network.getParamsValues())
            print(np.mean(paramsFromNetwork[0][0,0,:,:]))
            if len(params)>0:
                for i in range(len(params)):
                    print(params[i].split(':'))
                    param=params[i].split(':')
                    if(len(paramsFromNetwork[layer*2].shape)>2):
                        #CONV LAYER
                        if len(param)>1:
                            kernels=paramsFromNetwork[layer*2][int(param[0]):int(param[1]),0,:,:]
                        else:
                            kernels=paramsFromNetwork[layer*2][int(param[0]):int(param[0])+1,0,:,:]
                        print(kernels.shape)
                        for k in range(kernels.shape[0]):
                            export=[]
                            for i in range(kernels.shape[1]):
                                for j in range(kernels.shape[2]):
                                    export.append(kernels[k,i,j])
                            export=self.formatImage(export)
                            paramsList.append([int(param[0])+k,export])
                    else:
                        #FC LAYER ?
                        if len(param)>1:
                            kernels=paramsFromNetwork[layer*2][:,int(param[0]):int(param[1])]
                        else:
                            kernels=paramsFromNetwork[layer*2][:,int(param[0]):int(param[0])+1]
                        print(kernels.shape)
                        for k in range(kernels.shape[1]):
                            export=[]
                            for i in range(kernels.shape[0]):
                                export.append(kernels[i,k])
                            export=self.formatImage(export)
                            paramsList.append([int(param[0])+k,export])
                result=dict()
                result['layer']=layer
                result['paramsList']=paramsList
                return result
            else:
                return 0
        else:
            return 0
        
    def getPipeline(self,params):
        layerIndex=int(params['layer'])
        inputIndex=int(params['input'])
        inputImage=self.TRAIN_DATASET[0]
        paramsFromNetwork=np.asarray(self.network.getParamsValues())
        export=[]
        if len(paramsFromNetwork[layerIndex*2].shape)>2:
            #CONV
            for k in range(paramsFromNetwork[layerIndex*2].shape[0]):
                img=self.network.outputFunctions[layerIndex](inputImage[inputIndex:inputIndex+1])[0,k,:,:]
                res=[]
                for i in range(img.shape[0]):
                    for j in range(img.shape[1]):
                        res.append(img[i,j])
                res=self.formatImage(res)
                export.append([k,res])
        else:
            #FC
            img=self.network.outputFunctions[layerIndex](inputImage[inputIndex:inputIndex+1])
            print(img.shape)
            if img.shape[1]>200:
                for k in range(0,img.shape[1],200):
                    if k+200>img.shape[1]:
                        imgT=img[:,k:img.shape[1]]
                    else:
                        imgT=img[:,k:k+200]
                    print(imgT.shape)
                    res=[]
                    for i in range(imgT.shape[0]):
                        for j in range(imgT.shape[1]):
                            res.append(imgT[i,j])
                    res=self.formatImage(res)
                    export.append([k,res])
            else:
                res=[]
                for i in range(img.shape[0]):
                    for j in range(img.shape[1]):
                        res.append(img[i,j])
                res=self.formatImage(res)
                export.append([0,res])
        result=dict()
        result['layer']=layerIndex
        result['images']=export
        return result
    
    def formatImage(self,img):
        maxV=np.max(img)
        minV=np.abs(np.min(img))
        output=[]
        for pixel in img:
            pixel+=minV
            pixel/=(minV+maxV+0.000001)
            pixel*=255
            output.append(str(np.int(pixel)))
        return output
            
                