#Basically I just rewrite all the code the Qwen neural network wrote me, but I ask clarifying questions to understand everything and be able to recreate the code myself
import numpy as np
from skimage import io, color, transform

def conv2d(input,kernel):
    h,w=input.shape
    kh,kw=kernel.shape
    out_h=h-kh+1
    out_w=w-kw+1
    output=np.zeros((out_h,out_w))
    
    for i in range(out_h):
        for j in range(out_w):
            region=input[i:i+kh,j:j+kw]
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

def softmax(x):
    bananas=np.exp(x-np.max(x)) #chatgpt told me I can use "bananas" name instead of "exps"
    return bananas/np.sum(bananas)

def cross_entropy_loss(probs, label):
    return -np.log(probs[label]+1e-10)

def fully_connected(x,w,b):
    return np.dot(w,x)+b

def grad_fully_connected(x,w,b, probs, label):
    dz=probs.copy()
    dz[label]-=1
    dW=np.outer(dz,x)
    db=dz #hmm a nice advantage of using vectors instead of matrices is that you don't need to sum dZ (axis=0) when calculating db
    dx=np.dot(w.T, dz)
    return dW, db, dz

#load and preprocess image
def load_image(url, target_size=(28,28)):
    img=io.imread(url, as_gray=True)
    img=transform.resize(img, target_size, anti_aliasing=True)
    return img

url1='https://i.pinimg.com/736x/ae/21/16/ae21169a60da457b785d29438215174d.jpg'
url2='https://i.pinimg.com/736x/b8/3f/62/b83f622bffebc43930e8334c0b2f2e0b.jpg'
images=[load_image(url1), load_image(url2)]
labels=[3,7]
num_classes=10

#1 filter
kernel=np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
    ])

flat_size=((28-3+1)//2)*((28-3+1)//2)
fc_weights=np.random.randn(num_classes, flat_size)
fc_bias=np.zeros(num_classes)
learning_rate=0.01

#training epoch
for epoch in range(20):
    for idx in range (len(images)):
        image=images[idx]
        label=labels[idx]
        #forward pass
        conv_out=conv2d(image, kernel)
        relu_out=relu(conv_out)
        pool_out=max_pooling(relu_out)
        flat=pool_out.flatten()
        fc_out=fully_connected(flat, fc_weights, fc_bias)
        probs=softmax(fc_out)
        loss=cross_entropy_loss(probs,label)
        print (f"epoch {epoch}, loss {loss:.4f}, predicted {np.argmax(probs)}, actual {label}")
        #backward pass (only the fully connected layer for now)
        dW, db, dz=grad_fully_connected(flat, fc_weights, fc_bias, probs, label)
        #updates
        fc_weights-=dW*learning_rate
        fc_bias-=db*learning_rate
 
print ("Output probabilities:\n", probs)
print ("Predicted class:\n", np.argmax(probs))
