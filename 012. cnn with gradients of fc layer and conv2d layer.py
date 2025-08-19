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
    return np.maximum(0,x)

def max_pooling(x, size=2, stride=2):
    h,w=x.shape
    out_h=h//size
    out_w=w//size
    output=np.zeros((out_h,out_w))
    for i in range(out_h):
        for j in range (out_w):
            region=x[i*stride:i*stride+size, j*stride:j*stride+size]
            output[i,j]=np.max(region)
    return output

def flatten(x):
    return x.flatten()

def fully_connected(x, weight, bias):
    return np.dot(weight,x)+bias

def softmax(x):
    exps=np.exp(x-np.max(x)) #preventing numerical instability
    return exps/np.sum(exps)

def cross_entropy_loss(probs, label):
    return -np.log(probs[label]+1e-10)

#-----------------------BACKWARD PASS FUNCTIONS------------------------------------------------------

def grad_fully_connected(x,weights,probs,label):
    dlogits=probs.copy()
    dlogits[label]-=1
    dfc_weights=np.outer(dlogits,x)
    dfc_bias=dlogits
    dx=np.dot(weights.T,dlogits)
    return dfc_weights, dfc_bias, dx

def unflatten_gradient (flat_grad, shape=(13,13)):
    return flat_grad.reshape(shape)


def grad_max_pool (dpool_out, relu_out, size=2, stride=2):
    d_relu=np.zeros_like(relu_out)
    ph, pw=dpool_out.shape
    for i in range (ph):
        for j in range (pw):
            #get the region from the relu output
            region=relu_out[i*stride:i*stride+size,j*stride:j*stride+size]
            max_pos=np.unravel_index(np.argmax(region),region.shape)
            #set gradient only for the max position
            d_relu[i*stride+max_pos[0],j*stride+max_pos[1]]=dpool_out[i,j]
    return d_relu

def grad_relu(d_after_relu, pre_relu):
    d_relu=d_after_relu.copy()
    d_relu[pre_relu<=0]=0
    return d_relu

def grad_conv()
    
# CONTINUE TO DO DEBUGGING, i STOPPED AT dlogits=probs.copy()