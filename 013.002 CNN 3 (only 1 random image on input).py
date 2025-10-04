#I need to use 1-hot labels, because I use softmax and cross-entropy loss
#Input: 4 random images from each of the 4 categories => random.shuffle them => vertical stack => conv2d => a stack of (luckily) 4 one-hot labels
#Need to add np.random.permutation to choose a random gun array: perm = np.random.permutation(len(weapons))
#What if, to simplify the network, I'll feed it only 1 image (but random one from a random type), and make it to define its type? And not feed it an imagine with 4 random guns?
#I added 1 input image and 1 output label for this image, BUT it needs one-hot label! Like [0,1,0,0], not [0] or [1]. Or it will work? I need to think.
#Ok, I added the one-hot labels! Should work fine. Now I need to feed them to the network.
#I think I need to add more fc layers. The network just doesn't understand the task at all. Pretty much random result after each epoch.
#Need to do backpropagation through the 4 fc layers now. There should be different function definitions for fc layers. 1st (from top) includes gradient of softmax, but further fc layers shouldn't have the gradient of the softmax!
#Need many convolutional+pooling layers (as chatgpt adivced) or bette a multi-layered kernel! The program now works fine. Sometimes the convergence is very slow, but often it's very reliable and steady. But it has overfitting! It's noticable on the inference data.
#with many kernels in one layer it works and even generalizes well! 400 eopchs are enough. Though with mistakes. Added L2 regularization.

import numpy as np
import random
from skimage.io import imread, imsave
from skimage.transform import resize
import time

input_image_number_test=3
learned_in=1
i=0
loss=1
comp_size=64
input_image_number=1
hid_nrn=10
num_kernels=10
kernel=np.random.randn(num_kernels,3,3)*np.sqrt(2/(3*3)) #he init
num_classes=4
h_transformations=(comp_size-3+1)//2
w_transformations=(comp_size-3+1)//2
fc_x_dim=h_transformations*w_transformations
fc_in_dim=fc_x_dim*num_kernels
fc_w1=np.random.randn(hid_nrn, fc_in_dim)/np.sqrt(fc_in_dim)
fc_b1=np.zeros(hid_nrn)
fc_w2=np.random.randn(hid_nrn, hid_nrn)/np.sqrt(hid_nrn)
fc_b2=np.zeros(hid_nrn)
fc_w3=np.random.randn(hid_nrn, hid_nrn)/np.sqrt(hid_nrn)
fc_b3=np.zeros(hid_nrn)
fc_w4=np.random.randn(num_classes, hid_nrn)/np.sqrt(hid_nrn)
fc_b4=np.zeros(num_classes)
learning_rate=0.001
epochs=400

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

#forward pass functions
def conv2d_multi(image,kernel): #multi because  I added many kernels
    h,w=image.shape
    num_kernels,kh,kw=kernel.shape
    out_h=h-kh+1
    out_w=w-kw+1
    output=np.zeros((num_kernels,out_h,out_w))
    for k in range(num_kernels):
        for i in range(out_h):
            for j in range(out_w):
                region=image[i:i+kh, j:j+kw]
                output[k,i,j]=np.sum(region*kernel[k])
    return output
def relu(x):
    return np.maximum(0,x)
def max_pooling(x, size=2, stride=2):
    num_kernels,h,w=x.shape
    out_h = (h - size + stride) // stride #ChatGPT gave me a generic formula for this, that works for any size and stride, but it wasn't intuitive, so I changed it
    out_w = (w - size + stride) // stride
    output=np.zeros((num_kernels,out_h,out_w))
    for k in range(num_kernels):
        for i in range(out_h):
            for j in range (out_w):
                region=x[k,i*stride:i*stride+size, j*stride:j*stride+size]
                output[k,i,j]=np.max(region)
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
def grad_softmax_fully_connected(input_below,weights,probs,label):
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
    num_kernels,ph,pw=dpool_out.shape
    for k in range(num_kernels):
        for i in range (ph):
            for j in range (pw):
                #get the region from the relu output
                region=relu_out[k,i*stride:i*stride+size,j*stride:j*stride+size]
                max_pos=np.unravel_index(np.argmax(region),region.shape)
                #set gradient only for the max position
                d_relu[k,i*stride+max_pos[0],j*stride+max_pos[1]]=dpool_out[k,i,j] #no need for +=, because I don't overlap the pooling windows
    return d_relu
def grad_relu(d_after_relu, pre_relu):
    d=d_after_relu.copy()
    d[pre_relu<=0]=0
    return d
def grad_conv2d_multi(image, d_conv_out, kernel_shape):
    num_kernels,kh,kw=kernel_shape
    dkernel=np.zeros((num_kernels,kh,kw))
    dk,dh,dw=d_conv_out.shape
    for k in range (dk):
        for i in range (dh):
            for j in range (dw):
                region=image[i:i+kh,j:j+kw]
                dkernel[k]+=region*d_conv_out[k,i,j]
    return dkernel

start=time.time() #time variables
prev_time=time.time()

for epoch in range(epochs):
    if loss<0.5: #randomizing labels  and inputs (like in the beginning of the program)
        choice=random.randint(0,len(weapons)-1)
        image=weapons[choice]
        index=indices[choice]
        labels=([0]*4)
        labels[index]=1
        true_label=np.argmax(labels)
        print (f"epoch {epoch}, learned in {learned_in} epochs")
        print ("---")
        learned_in=0
    
    learned_in+=1

    #forward pass
    conv_out=conv2d_multi(image,kernel)
    relu_out=relu(conv_out)
    pool_out=max_pooling(relu_out,size=2,stride=2)
    flat=flatten(pool_out)
    z1=fully_connected(flat,fc_w1,fc_b1)
    z2=fully_connected(z1,fc_w2,fc_b2)
    z3=fully_connected(z2,fc_w3,fc_b3)
    z4=fully_connected(z3,fc_w4,fc_b4)
    probs=softmax(z4)
    main_loss=cross_entropy_loss(probs,true_label)
    loss=main_loss+0.001*np.sum(kernel**2) #L2 regularization

    #backward pass
    dfc_w4, dfc_b4, d_z3=grad_softmax_fully_connected(z3,fc_w4,probs,true_label)
    dfc_w3, dfc_b3, d_z2=grad_fully_connected(z2,fc_w3,d_z3)
    dfc_w2, dfc_b2, d_z1=grad_fully_connected(z1,fc_w2,d_z2)
    dfc_w1, dfc_b1, d_flat=grad_fully_connected(flat,fc_w1,d_z1)
    d_pool=unflatten_gradient(d_flat, pool_out.shape)
    d_relu_from_pool=grad_max_pool(d_pool, relu_out, size=2, stride=2)
    d_conv_out=grad_relu(d_relu_from_pool, conv_out)
    dkernel=grad_conv2d_multi(image, d_conv_out, kernel.shape)

    #sgd param update
    fc_w1-=dfc_w1*learning_rate
    fc_b1-=dfc_b1*learning_rate
    fc_w2-=dfc_w2*learning_rate
    fc_b2-=dfc_b2*learning_rate
    fc_w3-=dfc_w3*learning_rate
    fc_b3-=dfc_b3*learning_rate
    fc_w4-=dfc_w4*learning_rate
    fc_b4-=dfc_b4*learning_rate
    kernel-=dkernel*learning_rate

    if epoch%10==0 or epoch==epochs-1:
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

bows_test=[image_to_matrix(r"D:\Pictures\machine_learning_pictures\guns_data\bow_test_"+str(i)+".png") for i in range (input_image_number_test)] #a list of bows
pistols_test=[image_to_matrix(r"D:\Pictures\machine_learning_pictures\guns_data\pistol_test_"+str(i)+".png") for i in range (input_image_number_test)]
rifles_test=[image_to_matrix(r"D:\Pictures\machine_learning_pictures\guns_data\rifle_test_"+str(i)+".png") for i in range (input_image_number_test)]
shotguns_test=[image_to_matrix(r"D:\Pictures\machine_learning_pictures\guns_data\shotgun_test_"+str(i)+".png") for i in range (input_image_number_test)]
weapons_test=bows_test+pistols_test+rifles_test+shotguns_test

for i in range(5): #randomize inference input and test 5 times
    choice_test=random.randint(0,len(weapons_test)-1)
    image_test=weapons_test[choice_test]
    indices_test=([0]*len(bows_test)+[1]*len(pistols_test)+[2]*len(rifles_test)+[3]*len(shotguns_test))
    index_test=indices_test[choice_test]
    labels_test=([0]*4)
    labels_test[index_test]=1
    true_label_test=np.argmax(labels_test)

    conv_out=conv2d_multi(image_test,kernel)
    relu_out=relu(conv_out)
    pool_out=max_pooling(relu_out,size=2,stride=2)
    flat=flatten(pool_out)
    z1=fully_connected(flat,fc_w1,fc_b1)
    z2=fully_connected(z1,fc_w2,fc_b2)
    z3=fully_connected(z2,fc_w3,fc_b3)
    z4=fully_connected(z3,fc_w4,fc_b4)
    probs=softmax(z4)
    loss=cross_entropy_loss(probs,true_label_test)

    print (f"inference {i}")
    print(f"prediction: {np.argmax(probs)}")
    print(f"correct class: {np.argmax(labels_test)}")
    print(f"loss: {float(loss)}")
    print ("---")

