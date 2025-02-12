import numpy as np #program that predicts sum of 3 numbers
lr=0.000001
alpha=0.01
def leaky_relu(x):
    return np.where(x>0,x,alpha*x)
def leaky_relu_derivative(x):
    return np.where(x>0,1,alpha)

inputs=np.random.randn(200,3)
targets=np.sum(inputs, axis=1, keepdims=True)
#input is 200,3, target is 200,1
#print(f"inputs:\n{inputs}\ntargets:\n{targets}\n")

#weights and biases initialization
w1=np.random.randn(3,20)*np.sqrt(2/3)
w2=np.random.randn(20,20)*np.sqrt(2/20)
w3=np.random.randn(20,1)*np.sqrt(2/20)
b1=np.zeros((1,20))
b2=np.zeros((1,20))
b3=np.zeros((1,1))

epochs=2000
for epoch in range(epochs):
    #forward pass
    z1=(inputs@w1)+b1
    a1=leaky_relu(z1)
    z2=(a1@w2)+b2
    a2=leaky_relu(z2)
    z3=(a2@w3)+b3
    a3=z3

    #backpropagation, z3g-gradient of z3, w2g-gradient of w2, b1g-gradient of b1
    l=np.mean((a3-targets)**2)
    #layers
    z3g=2*(a3-targets)
    z2g=z3g@w3.T*leaky_relu_derivative(z2)
    z1g=z2g@w2.T*leaky_relu_derivative(z1)
    #weights
    w3g=a2.T@z3g
    w2g=a1.T@z2g
    w1g=inputs.T@z1g
    #biases
    b3g=np.sum(z3g, axis=0, keepdims=True)
    b2g=np.sum(z2g, axis=0, keepdims=True)
    b1g=np.sum(z1g, axis=0, keepdims=True)
    #updating
    #weights
    w3-=w3g*lr
    w2-=w2g*lr
    w1-=w1g*lr
    #biases
    b3-=b3g*lr
    b2-=b2g*lr
    b1-=b1g*lr
    #process check
    if epoch%(epochs/10)==0: print(f"epoch: {epoch}, MSE: {l}")
    if epoch==epochs-1: print(f"epoch: {epoch}, MSE: {l}")
    if l>1e-2 and epoch%10==0: lr+=0.06*lr

#implementation
t_inputs=([[0.1,0.2,0.3]])
tz1=(t_inputs@w1)+b1
ta1=leaky_relu(tz1)
tz2=(ta1@w2)+b2
ta2=leaky_relu(tz2)
tz3=(ta2@w3)+b3
ta3=tz3 #test output (test a3)
print(f"t_inputs:\n{t_inputs}\ntest output:\n{ta3}")

# It seems like the main problem was in the huge size of the layers compared to the batch size.
# So the program couldn't generalize and just remembered how to multiply the sample batch.
# When I made the batch size=200 and number of neurons per layer = 20, it became much smarter in terms of generalization.
# Batch size 7 and number of neurons 100 didn't work well.😄
# So today I used leaky_relu, though it did'nt help.😄
# I wanted to make a program with big amount of layers, but I tried to fix it, and the layers have gone.😄
# Maybe I'll create big ass layers NN later. Wanna learn dropout technique though.
