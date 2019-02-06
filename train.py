import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#np.random.seed(1)

df = pd.read_csv("mnist_train.csv")

data = df.values

labels=data[:,0]
labels=labels.reshape((labels.shape[0],1))
images=data[:,1:].T



label=np.zeros((1,labels.shape[0]))
for i in range(labels.shape[0]):
    if labels[i,0] == 5:
        label[0,i]=1

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





W1=np.random.randn(12,images.shape[0])*0.001
W2=np.random.randn(12,12)*0.001
W3=np.random.randn(1,12)*0.001
B1=np.zeros((12,1))
B2=np.zeros((12,1))
B3=np.zeros((1,1))
A0=images/255



def forward_propagate(A0,W1,W2,W3,B1,B2,B3):
    Z1=W1.dot(A0)+B1
    A1=sigmoid(Z1)
    Z2=W2.dot(A1)+B2
    A2=sigmoid(Z2)
    Z3=W3.dot(A2)+B3
    A3=sigmoid(Z3)

    return Z1,A1,Z2,A2,Z3,A3

m=A0.shape[1]
lr=0.00013
for i in range(5000):
    Z1,A1,Z2,A2,Z3,A3 = forward_propagate(A0,W1,W2,W3,B1,B2,B3)
    

    dB3=np.sum(A3-label,axis=1,keepdims=True)/m
    dW3=(A3-label).dot(A2.T)
    dB2=np.sum(np.multiply(W3.T.dot(A3-label),sigmoidgrad(Z2)),axis=1,keepdims=True)/m
    dW2=np.multiply(W3.T.dot(A3-label),sigmoidgrad(Z2)).dot(A1.T)
    dB1=np.sum(np.multiply(W2.T.dot(np.multiply(W3.T.dot(A3-label),sigmoidgrad(Z2))),sigmoidgrad(Z1)),axis=1,keepdims=True)
    dW1=np.multiply(W2.T.dot(np.multiply(W3.T.dot(A3-label),sigmoidgrad(Z2))),sigmoidgrad(Z1)).dot(A0.T)
    
    
    W1 = W1-lr*dW1
    W2 = W2-lr*dW2
    W3 = W3-lr*dW3
    B1 = B1-lr*dB1
    B2 = B2-lr*dB2
    B3 = B3-lr*dB3
    #Z1,A1,Z2,A2,Z3,A3p = forward_propagate(A0,W1,W2,W3,B1,B2,B3)
    #if i % 10 == 0:
    error=cross_entropy(A3,label)
    print(error)
    if error <=0.099:
        lr=0.0005
    if error <= 0.10 and error >= 0.099:
        lr=0.0003
   # if error <=0.28:
   #     lr=0.000002
   # if error <= 0.18:
   #     lr=0.0000009
   # if error <= 0.175:
   #     lr=0.00000001
np.save('W1',W1)
np.save('W2',W2)
np.save('W3',W3)
np.save('B1',B1)
np.save('B2',B2)
np.save('B3',B3)









