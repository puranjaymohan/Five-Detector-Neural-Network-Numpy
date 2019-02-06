import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from PIL import Image
#np.random.seed(9)

def sigmoid(x):
    return 1/(1+np.exp(-x))
    #return np.tanh(x)
    #return x * (x > 0)

def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data

image=load_image("five.png")
#image=np.load('one.npy')
print(image.shape)
A0=image.reshape((784,1))/255


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
    

Z1,A1,Z2,A2,Z3,A3 = forward_propagate(A0,W1,W2,W3,B1,B2,B3)

print(A3)
plt.title((A3>0.19))
plt.imshow(image,cmap='gray')
plt.show()















