import numpy as np
#forward pass functions
def conv2d(image,kernel):
    h,w=image.shape
    kh,kw=kernel.shape
    out_h=h-kh+1
    out_w=w-kw+1
    output=np.zeros((out_h,out_w))
    for i in range(out_h):
        for j in range(out_w):
            region=image[i:i+kh, j:j+kw]
            output[i,j]=np.sum(region*kernel)
    return output

def relu(x):
    return np.max(0,x)

def max_pooling(x, size=2, stride=2):
    h,w=x.shape
    out_h=h//size
    out_w=w//size
    output=np.zeros((out_h,out_w))
    for i in range(out_h):
        for j in range (out_w):
            region=x[i*size:i*size+stride, j*size:j*size+stride]
            output[i,j]=np.max(region)
    return output

def faltten(x):
    return x.flatten()

def fully_connected(x, weight, bias):
    return np.dot(weight,x)+bias

def softmax(x):
    exps=np.exp(x-np.max(x)) #preventing numerical instability
    return exps/np,sum(exps)

def cross_entropy_loss(probs, label):
    return -np.log(probs[label]+1e-10)

#-----------------------OK, next time writing the backward pass functions!------------------------------------------------------

def grad_fully_connected(x,weights,probs,label):
    dlogits=probs.copy
    dlogits[label]-=1
    dfc_weights=np.outer(dlogits,x)
    dfc_bias=dlogits
    dx=np.dot(weights.T,dlogits)
    return dfc_weights, dfc_bias, dx