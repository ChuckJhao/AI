#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pylab as plt


def relu(x):
    return np.maximum(0, x)



def sigmoid(x):
    return 1 / (1 + np.exp(-x))  

# define activation: softmax
def softmax(x):
    if x.ndim == 2:
        x = x.T # Transpose it
        x = x - np.max(x, axis=0)
        #print(x)
        y = np.exp(x) / np.sum(np.exp(x),axis=0)
        return y.T
    return np.exp(x) / np.sum(np.exp(x))

def tangent(x):
    return (np.exp(x)-np.exp(-x)) / (np.exp(x) + np.exp(-x))  

# define cross_entropy
#Hy′(y):=−∑iy′ilog(yi)
# t is one hot encoding, 1的地方才要加
def cross_entropy(y, t):
    return -np.sum(t*np.log(y + 1e-7))   #1e-7 to avoid causing -inf
        
# define cross_entropy
#t:target is label ,1的地方才要加(其實也是label標對的地方)
def cross_entropy_batch(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size) 
        y = y.reshape(1, y.size)  
        
    if t.size == y.size:  
        t = t.argmax(axis=1)  
        #print(t)
                 
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

