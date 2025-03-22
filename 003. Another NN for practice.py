import numpy as np
#the program that predicts doubling of a number
inputs=np.random.randn(50,1)
targets=inputs*2
learning_rate=0.0001
w1=np.random.randn(1,100)
b1=np.random.randn(1,100)
w2=np.random.randn(100,1)

def relu(x):
    return np.maximum(0,x)

def relu_derivative(x):
    return np.where(x>0,1,0)

max_epochs=1000
for epoch in range(max_epochs):
    #forward pass
    z1=(inputs@w1)+b1
    a1=relu(z1)
    z2=a1@w2
    a2=z2
    #backpropagation
    l=np.mean((a2-targets)**2)
    dl=2*(a2-targets)
    #backpropagation for layers
    z2_error=dl #dL/dz2=dL/da2*da2/dz2
    z1_error=(z2_error@w2.T)*relu_derivative(z1) #dL/dz1=z2_error*dz2/da1*da1/dz1
    #backpropagation for weights
    w2_error=a1.T@z2_error #dL/dw2=z2_error*dz2/dw2=z2_error*a1
    w1_error=inputs.T@z1_error
    b1_error=np.sum(z1_error, axis=0, keepdims=True) #dL/db1=z1_error*dz1/db1 
    #updating weights and biases
    w1-=w1_error*learning_rate
    w2-=w2_error*learning_rate
    b1-=b1_error*learning_rate
        
    #break if the loss function is small
    if(l<10e-5):
        print(f"The for loop's interrupted earlier. Epoch={epoch}, MSE={l}, BREAK")
        break
    #print epoch and loss
    if epoch%100==0:
        print(f"Epoch={epoch}, MSE={l}")
    #print if the epoch ended about in the end
    if epoch==max_epochs-1:
        print(f"The for loop is over (epoch {epoch}, MSE={l}), BREAK")
    #fix learning rate if it's too slow
    if(epoch%100==0 and l>10e-5): learning_rate*=1.01
    
test_inputs=np.array(([0.1],[0.2],[0.3]))
test_z1=(test_inputs@w1)+b1
test_a1=relu(test_z1)
test_z2=test_a1@w2
test_a2=test_z2
print(f"test inputs:\n{test_inputs}\ntest_outputs:\n{test_a2}")

#Not the ideal program, 
#but it does predictions stable and more or less confident, not some random numbers.
#The best thing that worked is in the line 48, where it increases learning rate,
#if it's slow.
#Did this program with mostly help from ChatGPT in debugging.

#Output:
#Epoch=0, MSE=4.053517306826315
#Epoch=100, MSE=0.11606078216723267
#Epoch=200, MSE=0.0450511622940421
#Epoch=300, MSE=0.027741059736178464
#Epoch=400, MSE=0.020492444227452243
#Epoch=500, MSE=0.016283424133747962
#Epoch=600, MSE=0.013289143934072933
#Epoch=700, MSE=0.0109678554509577
#Epoch=800, MSE=0.00925672420510409
#Epoch=900, MSE=0.008014682159818858
#The for loop is over (epoch 999, MSE=0.007011718469756479), BREAK        
#test inputs:
#[[0.1]
# [0.2]
# [0.3]]
#test_outputs:
#[[0.15031799]
# [0.33502054]
# [0.59781723]]
