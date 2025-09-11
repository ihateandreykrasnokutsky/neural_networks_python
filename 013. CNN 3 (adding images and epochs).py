#I need to use 1-hot labels, because I use softmax and cross-entropy loss
#input: 4 random images from each of the 4 categories => random.shuffle them => vertical stack => conv2d => a stack of (luckily) 4 one-hot labels
#need to add np.random.permutation to choose a random gun array: perm = np.random.permutation(len(weapons))

import numpy as np
import random
from skimage.io import imread, imsave
from skimage.transform import resize
import time

comp_size=500
input_image_number=20
kernel=np.random.randn(3,3)*0.01
num_classes=4
h_transformations=(comp_size*num_classes-3+1)//2
w_transformations=(comp_size-3+1)//2
fc_x_dim=h_transformations*w_transformations
fc_in_dim=fc_x_dim
fc_weights=np.random.randn(num_classes, fc_in_dim)*0.01
fc_bias=np.zeros(num_classes)
learning_rate=3

#additional functions
def image_to_matrix(pic_path,size=(comp_size,comp_size)):
    matrix=imread(pic_path,as_gray=True)
    matrix=resize(matrix,size,anti_aliasing=True,preserve_range=True)
    return matrix

bows=[image_to_matrix(r"D:\Pictures\machine_learning_pictures\guns_data\bow_"+str(i)+".png") for i in range (input_image_number)] #a list of bows
pistols=[image_to_matrix(r"D:\Pictures\machine_learning_pictures\guns_data\pistol_"+str(i)+".png") for i in range (input_image_number)]
rifles=[image_to_matrix(r"D:\Pictures\machine_learning_pictures\guns_data\rifle_"+str(i)+".png") for i in range (input_image_number)]
shotguns=[image_to_matrix(r"D:\Pictures\machine_learning_pictures\guns_data\shotgun_"+str(i)+".png") for i in range (input_image_number)]

#creating image stack and label stack
bow=bows[random.randint(0,input_image_number-1)]
pistol=pistols[random.randint(0,input_image_number-1)]
rifle=rifles[random.randint(0,input_image_number-1)]
shotgun=shotguns[random.randint(0,input_image_number-1)]
weapons=[bow,pistol,rifle,shotgun]
labels=[0,1,0,0]#1 for pistols
perm=np.random.permutation(len(weapons))
weapons_shuffled=[weapons[i] for i in perm]
labels_shuffled=[labels[i] for i in perm]
image=np.vstack(weapons_shuffled) #40x10
labels=np.hstack(labels_shuffled) #4x1
true_label=np.argmax(labels) #CE loss expects an index of the true label, not the array of all labels


print("Shuffled labels:", labels_shuffled)
print("True label:", true_label)

#forward pass functions
def conv2d(image,kernel): #38x8
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
def relu(x): #38x8
    return np.maximum(0,x)
def max_pooling(x, size=2, stride=2): #19x4
    h,w=x.shape
    out_h = (h - size + stride) // stride #ChatGPT gave me a generic formula for this, that works for any size and stride, but it wasn't intuitive, so I changed it
    out_w = (w - size + stride) // stride
    output=np.zeros((out_h,out_w))
    for i in range(out_h):
        for j in range (out_w):
            region=x[i*stride:i*stride+size, j*stride:j*stride+size]
            output[i,j]=np.max(region)
    return output
def flatten(x): # 76x
    return x.flatten()
def fully_connected(x, weight, bias):
    return np.dot(weight,x)+bias
def softmax(x):
    exps=np.exp(x-np.max(x)) #preventing numerical instability, e.g. e^-100 is better than e^100, because 1st is almost 0 (one hot), and 2nd is very high.
    return exps/np.sum(exps)
def cross_entropy_loss(probs, label):
    return -np.log(probs[label]+1e-10)

#backward pass functions
def grad_fully_connected(x,weights,probs,label):
    dlogits=probs.copy()
    dlogits[label]-=1
    dfc_weights=np.outer(dlogits,x)
    dfc_bias=dlogits
    dx=np.dot(weights.T,dlogits)
    return dfc_weights, dfc_bias, dx
def unflatten_gradient (flat_grad, shape): #ChatGPT changed the constant size (13,13) to the variable size
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
            d_relu[i*stride+max_pos[0],j*stride+max_pos[1]]+=dpool_out[i,j]
    return d_relu
def grad_relu(d_after_relu, pre_relu):
    d=d_after_relu.copy()
    d[pre_relu<=0]=0
    return d
def grad_conv(image, d_conv_out, kernel_shape):
    dkernel=np.zeros(kernel_shape)
    kh,kw=kernel_shape
    dh,dw=d_conv_out.shape
    for i in range (dh):
        for j in range (dw):
            region=image[i:i+kh,j:j+kw]
            dkernel+=region*d_conv_out[i,j]
    return dkernel

#Tiny training demo
#forward pass
start=time.time()
print ("Start time: ", start)
conv_out=conv2d(image,kernel)
print ("conv2d time: ", time.time()-start)
relu_out=relu(conv_out)
print ("relu time: ", time.time()-start)
pool_out=max_pooling(relu_out,size=2,stride=2)
print ("max_pooling time: ", time.time()-start)
flat=flatten(pool_out)
print ("flatten time: ", time.time()-start)
logits=fully_connected(flat,fc_weights,fc_bias)
print ("fully_connected time: ", time.time()-start)
probs=softmax(logits)
print ("softmax: ", time.time()-start)
loss=cross_entropy_loss(probs,true_label)
print ("cross_entropy_loss time: ", time.time()-start)

print("Initial prediction: ", np.argmax(probs))
print("Loss: ", float(loss))

#backward pass
dfc_W, dfc_b, d_flat=grad_fully_connected(flat, fc_weights, probs, true_label)
d_pool=unflatten_gradient(d_flat, pool_out.shape)
d_relu_from_pool=grad_max_pool(d_pool, relu_out, size=2, stride=2)
d_conv_out=grad_relu(d_relu_from_pool, conv_out)
dkernel=grad_conv(image, d_conv_out, kernel.shape)

#sgd param update
fc_weights-=dfc_W*learning_rate
fc_bias-=dfc_b*learning_rate
kernel-=dkernel*learning_rate

#re-forward
conv_out=conv2d(image,kernel)
relu_out=relu(conv_out)
pool_out=max_pooling(relu_out,size=2,stride=2)
flat=flatten(pool_out)
logits=fully_connected(flat,fc_weights,fc_bias)
probs=softmax(logits)
loss=cross_entropy_loss(probs,true_label)

print ("After one update:")
print ("Prediction: ", np.argmax(probs))
print ("Loss: ", float(loss))

