import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join

def formatData(file):
    f=open(file,'r')
    content=f.readlines()
    f.close()
    values=[]
    error=[]
    for cont in content:
        if cont.find(';')>0:
            cont=cont.split(';')
            values.append([int(cont[0]),float(cont[1]),float(cont[2].split('\n')[0])])
        else:
            error.append(float(cont))
    nbEpochs=int(len(values)/len(error))
    values=np.matrix(values)
    error=np.matrix(error)
    f=open(file.split('.txt')[0]+'_export.txt','w')
    for i in range(nbEpochs):
        val=np.mean(values[np.where(values[:,0]==i)[0],:],axis=0)
        f.write(str(int(val[0,0]))+";"+str(val[0,1])+";"+str(val[0,2])+"\n")
    f.write(str(np.mean(error))+"\n")
    f.close()

def printDatas(filename):
    onlyfiles = [f for f in listdir(filename) if isfile(join(filename, f))]
    validation=[]
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    for file in onlyfiles:
        print(file)
        if(len(file.split('network'))>1):
            print(file.split('.txt')[0]+'_data_export.txt')
            f=open(file.split('.txt')[0]+'_data_export.txt','r')
            
            content=f.readlines()
            epochs=[]
            loss=[]
            y=[]
            for cont in content:
                if cont.find(';')>0:
                    cont=cont.split('\n')[0]
                    cont=cont.split(';')
                    epochs.append(cont[0])
                    loss.append(cont[1])
                    y.append(cont[2])
            ax1.scatter(epochs, y, s=10,  marker="s", label=file.split('_data_export.txt')[0])
    plt.legend(loc='upper right');

    plt.show()

def printData(filename):
    f=open(filename,'r')
    content=f.readlines()
    epochs=[]
    loss=[]
    validation=[]
    for cont in content:
        if cont.find(';')>0:
            cont=cont.split('\n')[0]
            cont=cont.split(';')
            print(cont[1])
            epochs.append(cont[0])
            loss.append(cont[1])
            validation.append(cont[2])
    print(validation)
    plt.plot(epochs,validation,'ro')
    plt.show()

printDatas('networks')   
