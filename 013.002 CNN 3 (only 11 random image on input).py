#I need to use 1-hot labels, because I use softmax and cross-entropy loss
#Input: 4 random images from each of the 4 categories => random.shuffle them => vertical stack => conv2d => a stack of (luckily) 4 one-hot labels
#Need to add np.random.permutation to choose a random gun array: perm = np.random.permutation(len(weapons))
#What if, to simplify the network, I'll feed it only 1 image (but random one from a random type), and make it to define its type? And not feed it an imagine with 4 random guns?
#I added 1 input image and 1 output label for this image, BUT it needs one-hot label! Like [0,1,0,0], not [0] or [1]. Or it will work? I need to think.
#Ok, I added the one-hot labels! Should work fine. Now I need to feed them to the network.
#I think I need to add more fc layers. The network just doesn't understand the task at all. Pretty much random result after each epoch.
#Need to do backpropagation through the 4 fc layers now. There should be different function definitions for fc layers. 1st (from top) includes gradient of softmax, but further fc layers shouldn't have the gradient of the softmax!

import numpy as np
import random
from skimage.io import imread, imsave
from skimage.transform import resize
import time

i=0
loss=0
comp_size=64
input_image_number=1
hid_nrn=50
kernel=np.random.randn(3,3)*0.01
num_classes=4
h_transformations=(comp_size-3+1)//2
w_transformations=(comp_size-3+1)//2
fc_x_dim=h_transformations*w_transformations
fc_in_dim=fc_x_dim
fc_w_1=np.random.randn(hid_nrn, fc_in_dim)*0.01
fc_b_1=np.zeros(hid_nrn)
fc_w_2=np.random.randn(hid_nrn, hid_nrn)*0.01
fc_b_2=np.zeros(hid_nrn)
fc_w_3=np.random.randn(hid_nrn, hid_nrn)*0.01
fc_b_3=np.zeros(hid_nrn)
fc_w_4=np.random.randn(num_classes, hid_nrn)*0.01
fc_b_4=np.zeros(num_classes)
learning_rate=1
epochs=10000

#additional functions
def image_to_matrix(pic_path,size=(comp_size,comp_size)):
    matrix=imread(pic_path,as_gray=True)
    matrix=resize(matrix,size,anti_aliasing=True,preserve_range=True)
    return matrix

bows=[image_to_matrix(r"D:\Pictures\machine_learning_pictures\guns_data\bow_"+str(i)+".png") for i in range (input_image_number)] #a list of bows
pistols=[image_to_matrix(r"D:\Pictures\machine_learning_pictures\guns_data\pistol_"+str(i)+".png") for i in range (input_image_number)]
rifles=[image_to_matrix(r"D:\Pictures\machine_learning_pictures\guns_data\rifle_"+str(i)+".png") for i in range (input_image_number)]
shotguns=[image_to_matrix(r"D:\Pictures\machine_learning_pictures\guns_data\shotgun_"+str(i)+".png") for i in range (input_image_number)]
weapons=bows+pistols+rifles+shotguns
indices=([0]*len(bows)+[1]*len(pistols)+[2]*len(rifles)+[3]*len(shotguns)) #indices for each weapon picture!
choice=random.randint(0,len(weapons)-1) #random choice for both weapons and indices
image=weapons[choice] #random picture (64,64)
index=indices[choice] #0/1/2/3
labels=([0]*4) #virgin labels [0,0,0,0], a row vector
labels[index]=1 #one-hot labels with added true label, so it should be like [0,0,1,0]
true_label=np.argmax(labels) #or just 1

print(f"indices={indices}")
print(f"choice={choice}")
print(f"index={index}")
print(f"labels={labels}")


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
    out_h = (h - size + stride) // stride #ChatGPT gave me a generic formula for this, that works for any size and stride, but it wasn't intuitive, so I changed it
    out_w = (w - size + stride) // stride
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
    exps=np.exp(x-np.max(x)) #preventing numerical instability, e.g. e^-100 is better than e^100, because 1st is almost 0 (one hot), and 2nd is very high.
    return exps/np.sum(exps)
def cross_entropy_loss(probs, label):
    return -np.log(probs[label]+1e-10)

#backward pass functions
def grad_fully_connected_last(input_below,weights,probs,label):
    dlogits=probs.copy()
    dlogits[label]-=1
    dfc_weights=np.outer(dlogits,input_below)
    dfc_bias=dlogits
    dx=np.dot(weights.T,dlogits)
    return dfc_weights, dfc_bias, dx

def grad_fully_connected(input_below,weights,grad_above):
    dfc_weights=np.outer(grad_above,input_below)
    dfc_bias=grad_above
    dx=np.dot(weights.T,grad_above)
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

start=time.time() #time variables
prev_time=time.time()

for epoch in range(epochs):
    if loss<0.1: #randomizing labels  and inputs (like in the beginning of the program)
        choice=random.randint(0,len(weapons)-1)
        image=weapons[choice]
        index=indices[choice]
        labels=([0]*4)
        labels[index]=1
        true_label=np.argmax(labels)

    #forward pass
    conv_out=conv2d(image,kernel)
    relu_out=relu(conv_out)
    pool_out=max_pooling(relu_out,size=2,stride=2)
    flat=flatten(pool_out)
    logits1=fully_connected(flat,fc_w_1,fc_b_1)
    logits2=fully_connected(logits1,fc_w_2,fc_b_2)
    logits3=fully_connected(logits2,fc_w_3,fc_b_3)
    logits4=fully_connected(logits3,fc_w_4,fc_b_4)
    probs=softmax(logits4)
    loss=cross_entropy_loss(probs,true_label)

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

    if epoch%100==0 or epoch==epochs-1:
        print(f"epoch {epoch}:")
        print (f"time: {time.time()-prev_time}")
        print(f"prediction: {np.argmax(probs)}")
        print(f"correct class: {np.argmax(labels)}")
        print(f"loss: {float(loss)}")
        print ("---")
        prev_time=time.time()


#inference
print ("inference:")
print ("---")

for i in range(5): #randomize inference input and test 5 times
    choice=random.randint(0,len(weapons)-1)
    image=weapons[choice]
    index=indices[choice]
    labels=([0]*4)
    labels[index]=1
    true_label=np.argmax(labels)

    conv_out=conv2d(image,kernel)
    relu_out=relu(conv_out)
    pool_out=max_pooling(relu_out,size=2,stride=2)
    flat=flatten(pool_out)
    logits=fully_connected(flat,fc_weights,fc_bias)
    probs=softmax(logits)
    loss=cross_entropy_loss(probs,true_label)

    print (f"inference {i}")
    print(f"prediction: {np.argmax(probs)}")
    print(f"correct class: {np.argmax(labels)}")
    print(f"loss: {float(loss)}")
    print ("---")

