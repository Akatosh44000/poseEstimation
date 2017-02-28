import numpy as np
import matplotlib.pyplot as plt

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
    
