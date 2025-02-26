import numpy as np

lr=0.001 #basic variables
epochs=1000

x=np.random.randn(10,1) #training input and output
y=x*2

w1=np.random.randn(1,40) #weights and biases
b1=np.zeros((1,40))
w2=np.random.randn(40,1)
b2=np.zeros((1,1))

for epoch in range (epochs):
    z1=x@w1+b1 #forward pass
    a1=z1
    z2=a1@w2+b2
    a2=z2
    
    l=np.mean((a2-y)**2)#loss
    z2grad=2*(a2-y) #gradients for normal update
    z1grad=z2grad@w2.T
    
    w2grad=a1.T@z2grad #weights and biases gradients
    w1grad=x.T@z1grad
    b2grad=np.sum(z2grad,axis=0,keepdims=True)
    b1grad=np.sum(z1grad) #don't collapse along 0 axis, because it's already collapsed
    
    w2-=w2grad*lr
    w1-=w1grad*lr
    b2-=b1grad*lr
    b1-=b1grad*lr
    
    if epoch%(epochs/5)==0 or epoch+1==epochs: #print epochs and MSE periodically
        print(f"epoch {epoch}, MSE {l}")
    if l<1e-8:
        print(f"epoch {epoch}, MSE {l}, BREAK")
        break

tx=np.random.randint(low=-100,high=100,size=(3,1)) #test input, applying w and b and printing output
tz1=tx@w1+b1
ta1=tz1
tz2=ta1@w2+b2
ta2=tz2
print (f"test input:\n{tx}\ntest output:\n{ta2}")
