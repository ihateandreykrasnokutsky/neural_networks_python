import numpy as np

alpha=0.01
def leaky_relu(x):
    return np.where(x>0,x,alpha*x)
def leaky_relu_derivative(x):
    return np.where(x>0,x,alpha)#d/dx(alpha*x)=alpha

rho=0.05 #basic variables
lr=0.01
epochs=1000

x=np.random.randn(10,1) #training input and output
y=x*2

w1=np.random.randn(1,10) #weights and biases
b1=np.zeros((1,10))
w2=np.random.randn(10,1)
b2=np.zeros((1,1))

for epoch in range (epochs):
    z1=x@w1+b1 #forward pass
    a1=leaky_relu(z1)
    z2=a1@w2+b2
    a2=z2
    
    l=np.mean((a2-y)**2)#loss
    z2grad=2*(a2-y) #gradients for normal update
    z1grad=z2grad@w2.T*leaky_relu_derivative(z1)
    
    w2grad=a1.T@z2grad #weights and biases gradients
    w1grad=x.T@z1grad
    b2grad=np.sum(z2grad,axis=0,keepdims=True)
    b1grad=np.sum(z1grad) #don't collapse along 0 axis, because it's already collapsed
    
    w2-=w2grad*lr
    w1-=w1grad*lr
    b2-=b1grad*lr
    b1-=b1grad*lr
    
    if epoch%(epochs/10)==0 or epoch+1==epochs: #print epochs and MSE periodically
        print(f"epoch {epoch}, MSE {l}")
    if l<1e-4:
        print(f"epoch {epoch}, MSE {l}, BREAK")
        break

tx=np.array([[1],[2],[3],[-1],[-2],[-3]])#test input, applying w and b and printing output
tz1=tx@w1+b1
ta1=leaky_relu(tz1)
tz2=ta1@w2+b2
ta2=tz2
print (f"test input:\n{tx}\ntestoutput:\n{ta2}")
