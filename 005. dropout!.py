#creating a neural network that uses dropout technique, works acceptable
import numpy as np

#define activation dunctions and their derivatives
def relu(x):
    return np.maximum(0,x)
def relu_derivative(x):
    return np.where(x>0,1,0)
def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_derivative(x):
    sig=sigmoid(x)
    return sig*(1-sig)
alpha=0.01
def leaky_relu(x):
    return np.where(x>0,x,alpha*x)
def leaky_relu_derivative(x):
    return np.where(x>0,1,alpha)
#define dropout function
def dropout(x,dropout_rate=0.001):
    mask=(np.random.rand(*x.shape)>dropout_rate)/(1-dropout_rate)
    return x*mask, mask

#assigments to easily switch between functions and derivatives and see how they affect the result
#activation=relu
#activation_derivative=relu_derivative
#activation=sigmoid
#activation_derivative=sigmoid_derivative
activation=leaky_relu
activation_derivative=leaky_relu_derivative

x=np.array([[-3.1],[-2.9],[-3],[0],[4],[3.0]])
y=x*2
lr=0.001
w1=np.random.randn(1,100)
b1=np.zeros((1,100))
w2=np.random.rand(100,1)
b2=np.zeros((1,1))

epochs=100000
for epoch in range(epochs):
    #forward pass
    z1=x@w1+b1
    a1=activation(z1)
    a1_drop,mask=dropout(a1)
    z2=a1_drop@w2+b2
    a2=z2
    #backpropagation
    #loss
    l=np.mean((a2-y)**2)
    #layer gradient
    z2grd=(2*(a2-y))
    z1grd=(z2grd@w2.T)*activation_derivative(z1)*mask
    #weight gradient
    w2grd=a1_drop.T@z2grd
    w1grd=x.T@z1grd
    #bias gradient
    b2grd=np.sum(z2grd, axis=0, keepdims=True)
    b1grd=np.sum(z1grd, axis=0, keepdims=True)
    #updates
    w2-=w2grd*lr
    w1-=w1grd*lr
    b2-=b2grd*lr
    b1-=b1grd*lr
    #output of the process
    if epoch%(epochs/10)==0:
        print(f"epoch={epoch}, MSE={l}")
    if l<1e-5:
        print(f"epoch={epoch}, MSE={l}, BREAK")
        break
    
#inference
tx=np.array([[-1],[-2],[3],[4],[5],[7]])
tz1=tx@w1+b1
ta1=activation(tz1)
tz2=ta1@w2+b2
ta2=tz2
print(f"tx:\n{tx}\nta2:\n{ta2}\n")

#OUTPUTS
#epoch=0, MSE=3006.1686759306954
#epoch=10000, MSE=0.08979964750145313
#epoch=20000, MSE=6.75633920845781e-05
#epoch=30000, MSE=0.00010624546785150009
#epoch=40000, MSE=0.0001281815523300913
#epoch=50000, MSE=0.03475703876413515
#epoch=52863, MSE=5.525596496286511e-06, BREAK
#tx:
#[[-1]
# [-2]
# [ 3]
# [ 4]
# [ 5]
# [ 7]]
#ta2:
#[[-1.58338648]
# [-3.79119091]
# [ 5.98984161]
# [ 7.99505432]
# [10.00026704]
# [14.01069248]]
