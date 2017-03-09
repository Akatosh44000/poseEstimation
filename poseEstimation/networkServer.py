'''
@author: j.langlois
'''
import numpy as np
import threading
import os
from socketIO_client import SocketIO,BaseNamespace
import network_conv_estimation
import import2D
import theano
import theano.tensor as T
import lasagne

class networkServer:
    
    def __init__(self):
        print('WAIT:: STARTING NETWORK CLIENT')

        #BYPASS PROXY RULES FOR LOCALHOST DOMAIN
        os.environ['NO_PROXY'] = 'localhost'
        
        #NETWORK PARAMS
        self.id=0
        self.name='reseau_1'
        
        self.importDataset()
        self.buildNetwork()
        
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
        self.network=network_conv_estimation.Network(self.TRAIN_DATASET[0].shape[1],
                                                     self.TRAIN_DATASET[0].shape[2],
                                                     self.TRAIN_DATASET[1].shape[1])
        self.network._observers.append(self.newEpoch)
        self.layers=lasagne.layers.get_all_layers(self.network.last_layer)
        print('SUCCESS:: NETWORK BUILT !')
        
    def handle_message_from_server(self,message):   
        if self.id==0:
            print('INFO:: NETWORK GOT THE ID '+message['network_id']+' FROM THE SERVER.')
            self.id=int(message['network_id']) 
            
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
            self.sendMessageToClient('params',{'params':self.getParams(request['params'])},request['client_socket_id']);
        if request['name']=='getPipeline':
            self.sendMessageToClient('pipeline',{'pipeline':self.getPipeline(request['params'])},request['client_socket_id']);
        if request['name']=='getDatasetParams':
            self.sendMessageToClient('datasetParams',{'datasetParams':self.getDatasetsParameters()},request['client_socket_id']);
            
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
        
    def newEpoch(self):
        print("INFO:: NEW EPOCH")
        self.sendMessageToserver('newEpoch','')
        
    def getNetworkArchitecture(self):
        architecture=[]
        print(self.layers)
        for layer in self.layers:
            if(str(layer.__class__).find('conv')>0):
                architecture.append(['CONV',layer.W.get_value().shape])
            elif(str(layer.__class__).find('dense')>0):
                architecture.append(['FC',layer.W.get_value().shape])
            elif(str(layer.__class__).find('pool')>0):
                architecture.append(['POOL',0])
            elif(str(layer.__class__).find('input')>0):
                architecture.append(['INPUT',self.TRAIN_DATASET[0].shape[1:]])
            else:
                architecture.append(['UNKNOWN',0])
        return architecture
    
    def getParams(self,parameters):
        if len(parameters)>0:
            paramsList=[]
            params=parameters['params']
            layerIndex=parameters['layer']
            layer=self.layers[layerIndex]
            layerParams=layer.W.get_value()
            print(np.mean(layerParams))
            if len(params)>0:
                for i in range(len(params)):
                    print(params[i].split(':'))
                    param=params[i].split(':')
                    if str(layer.__class__).find('conv')>0:
                        #CONV LAYER
                        if len(param)>1:
                            kernels=layerParams[int(param[0]):int(param[1]),0,:,:]
                        else:
                            kernels=layerParams[int(param[0]):int(param[0])+1,0,:,:]
                        print(kernels.shape)
                        for k in range(kernels.shape[0]):
                            export=[]
                            for i in range(kernels.shape[1]):
                                for j in range(kernels.shape[2]):
                                    export.append(kernels[k,i,j])
                            export=self.formatImage(export)
                            paramsList.append([int(param[0])+k,export])
                    elif str(layer.__class__).find('dense')>0:
                        #FC LAYER ?
                        if len(param)>1:
                            kernels=layerParams[:,int(param[0]):int(param[1])]
                        else:
                            kernels=layerParams[:,int(param[0]):int(param[0])+1]
                        print(kernels.shape)
                        for k in range(kernels.shape[1]):
                            export=[]
                            for i in range(kernels.shape[0]):
                                export.append(kernels[i,k])
                            export=self.formatImage(export)
                            paramsList.append([int(param[0])+k,export])
                result=dict()
                result['layer']=layerIndex
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
        layer=self.layers[layerIndex]
        input_var = T.tensor4('inputs')     
        getOutput=theano.function([input_var],lasagne.layers.get_output(layer,input_var))
        imgs=getOutput(inputImage[inputIndex:inputIndex+1])
        
        export=[]
        if str(layer.__class__).find('conv')>0 or str(layer.__class__).find('pool')>0 or str(layer.__class__).find('input')>0:
            #CONV
            for k in range(imgs.shape[1]):
                img=imgs[0,k,:,:]
                res=[]
                for i in range(img.shape[0]):
                    for j in range(img.shape[1]):
                        res.append(img[i,j])
                res=self.formatImage(res)
                export.append([k,res])
        
        elif str(layer.__class__).find('dense')>0:
            #FC
            img=imgs
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
    
    def getDatasetsParameters(self):
        result=dict()
        if self.TRAIN_DATASET:
            set=dict()
            set['name']='TRAINING'
            set['params']=self.TRAIN_DATASET[0].shape
            result['training']=set
        if self.VALIDATION_DATASET:
            set=dict()
            set['name']='VALIDATION'
            set['params']=self.VALIDATION_DATASET[0].shape
            result['validation']=set  
        if self.TEST_DATASET:
            set=dict()
            set['name']='TEST'
            set['params']=self.TEST_DATASET[0].shape
            result['test']=set
        return result       
                