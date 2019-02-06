import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
#np.random.seed(9)

dftrain = pd.read_csv("mnist_train.csv")
dftest = pd.read_csv("mnist_test.csv")
datatrain = dftrain.values
datatest = dftest.values

labelstrain=datatrain[:,0]
labelstest=datatest[:,0]
labelstrain=labelstrain.reshape((labelstrain.shape[0],1))
labelstest=labelstest.reshape((labelstest.shape[0],1))
imagestrain=datatrain[:,1:].T
imagestest=datatest[:,1:].T

np.save("one",imagestrain[:,8].reshape((28,28)))

labeltrain=np.zeros((1,labelstrain.shape[0]))
for i in range(labelstrain.shape[0]):
    if labelstrain[i,0] == 5:
        labeltrain[0,i]=1

labeltest=np.zeros((1,labelstest.shape[0]))
for i in range(labelstest.shape[0]):
    if labelstest[i,0] == 5:
        labeltest[0,i]=1


def sigmoid(x):
    return 1/(1+np.exp(-x))
    #return np.tanh(x)
    #return x * (x > 0)
def sigmoidgrad(x):
    s=sigmoid(x)
    return s*(1-s)
    #return 1-sigmoid(x)**2
    #return 1. * (x > 0)

def cross_entropy(X,y):    
    m=X.shape[1]
    loss=(1/m)*np.sum((-(y*(np.log(X))+(1-y)*(np.log(1-X))))) 
    return loss





W1=np.load("W1.npy")
W2=np.load("W2.npy")
W3=np.load("W3.npy")
B1=np.load("B1.npy")
B2=np.load("B2.npy")
B3=np.load("B3.npy")




def forward_propagate(A0,W1,W2,W3,B1,B2,B3):
    Z1=W1.dot(A0)+B1
    A1=sigmoid(Z1)
    Z2=W2.dot(A1)+B2
    A2=sigmoid(Z2)
    Z3=W3.dot(A2)+B3
    A3=sigmoid(Z3)

    return Z1,A1,Z2,A2,Z3,A3
    
A0train=imagestrain/255
A0test=imagestest/255

Z1,A1,Z2,A2,Z3,A3train = forward_propagate(A0train,W1,W2,W3,B1,B2,B3)
A3train=np.where(A3train>0.3,1,0)
Z1,A1,Z2,A2,Z3,A3test = forward_propagate(A0test,W1,W2,W3,B1,B2,B3)
A3test=np.where(A3test>0.3,1,0)


predicttrain=A3train[0,:]
predicttest=A3test[0,:]
realtrain=labeltrain[0,:]
realtest=labeltest[0,:]

print("Train Accuracy :",accuracy_score(realtrain,predicttrain))
print("Test Accuracy :",accuracy_score(realtest,predicttest))

