import numpy as np
#the NN that calculates a/b, the NN also has 100 hidden neurons, it will teach me a bit to work with dimensions of matrices
def relu(x):
    return np.maximum(0,x)
def relu_derivative(x):
    return np.where(x>0,1,0)

inputs=np.array([[1,4],[4,2],[3,3],[5,2],[3,2],[1,2],[0,3],[1,5],[1,8],[1,10]])#10 examples, 2 features on each
targets=np.array([[0.25],[2],[1],[2.5],[1.5],[0.5],[0],[0.2],[0.125],[0.1]])#10 target outputs, 1 feature in each
learning_rate=0.00001

w1=np.random.rand(2,100)
b1=np.random.rand(1,100)
w2=np.random.rand(100,1)
b2=np.random.rand(1,1)

for epoch in range(1000):
    #forward pass
    z1=(inputs@w1)+b1
    a1=relu(z1)
    z2=(a1@w2)+b2
    a2=relu(z2)
    #backpropagation
    #loss function
    l=np.mean((a2-targets)**2)
    #error contribution of each layer
    z2_error=2*(a2-targets)*relu_derivative(z2) #dl/dz2=(dl/da2)*(da2/dz2)
    z1_error=(z2_error@w2.T)*relu_derivative(z1) #(dl/da2)*(da2/dz2)*(dz2/da1)*(da1/dz1)=l' * z2' * w2* z1'
    #error contribution of each weight
    w2_error=a1.T@z2_error #dl/dw2=z2_error*(dz2/dw2)=z2_error*a1
    w1_error=inputs.T@z1_error #dl/dw1=dz1_error*(dz1/dw1)=z1_error*inputs
    #error contribution of each bias
    #b2: dL/db2=(dL/da2)*(da2/z2)*(dz2/db2)=z2_error*1 (since w2 and a1 are constants, and 1*b1 is a variable)
    #b1: dL/db1=z1_error*(dz1/db1)=z1_error*1
    b2_error=np.sum(z2_error,axis=0,keepdims=True)
    b1_error=np.sum(z1_error,axis=0,keepdims=True)
    #applying errors*learning rate to weights and biases
    w1-=learning_rate*w1_error
    b1-=learning_rate*b1_error
    w2-=learning_rate*w2_error
    b2-=learning_rate*b2_error
    if epoch%100==0: print(f"epoch {epoch}, L={l}")
    if l<10e-2: break
    
test_inputs=np.array([[10,5],[16,4],[20,4], [1,1]])
test_z1=(test_inputs@w1)+b1
test_a1=relu(test_z1)
test_z2=(test_a1@w2)+b2
test_a2=relu(test_z2)
print(f"test input is \n{test_inputs}\n test output is \n{test_a2}")

#the output
#epoch 0, L=38764.24205499281
#epoch 100, L=0.22505914014980638
#epoch 200, L=0.21967845915388945
#epoch 300, L=0.21464981925689738
#epoch 400, L=0.2097952116120184
#epoch 500, L=0.2049951994361491
#epoch 600, L=0.20038644660995247
#epoch 700, L=0.19610788653704556
#epoch 800, L=0.1921356812148598
#epoch 900, L=0.18844774924849184
#test input is 
#[[10  5]
# [16  4]
# [20  4]
# [ 1  1]]
# test output is
#[[ 0.        ]
# [ 7.92053668]
# [12.79297113]
# [ 2.8954091 ]]
#PS C:\Users\Andrey Krasnokutsky\Documents\programming\python> 
#my thoughts: lower learning rate doesnt work, higher or lower amount of epochs dont work too, it seems like the program doesn't see that it does mistakes and actully is farther away from the correct result than it thinks
