import lasagne
import sys

def importFromFile(fileName):
    f=open(fileName,'r')
    content=f.readlines()
    layers=[]
    for layer in content:
        layer=layer.split('\n')[0]
        layer=layer.split(' ')
        params=layer[1:]
        for i in range(len(params)):
            params[i]=int(params[i])
        layers.append([layer[0],params])
    return layers
    
def createLayers(fileName,net):
    layers=importFromFile('networks/network1.txt')
    print("CREATING NETWORK ARCHITECTURE ...")
    print("INPUT --> ",end='')
    for layer in layers:
        print(layer,end='')
        print(" --> ",end='')
        net=createLayer(layer[0],net,layer[1])
    print("OUTPUT")
    print("ARCHITECTURE CREATED.")
    return net

def createLayer(layer_type,previous_layer,params):
    if layer_type=='CONV':
        return lasagne.layers.Conv2DLayer(previous_layer, num_filters=params[2], filter_size=(params[0], params[1]),nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.GlorotUniform())    
    elif layer_type=='POOL':
        return lasagne.layers.MaxPool2DLayer(previous_layer, pool_size=(params[0], params[1])) 
    elif layer_type=='FC':
        return lasagne.layers.DenseLayer(previous_layer,num_units=params[0],nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.Orthogonal())
    
