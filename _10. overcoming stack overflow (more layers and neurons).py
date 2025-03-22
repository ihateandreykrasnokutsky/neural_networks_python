#I wanted this program to be modified to add computation graphs
#(in the form of class that will include forward pass, backpropagation and other operations),
#but it turned out such modification will change about 90% of the code,
#so there's no sense in modifying it, it's better to just create a new code.
#The program predicts a sum of 2 numbers, while having a lot of layers and neurons.
#Nothing sepcial, including the performance,
#but I've learned how to increase number of neurons and layers without facing stack overflow.
import numpy as np

#define functions
def leaky_relu(x, alpha=0.05): #activation functions
    return np.where(x>0,x,alpha*x)
def leaky_relu_derivative(x, alpha=0.05): #derivative of the activation
    return np.where(x>0, 1, alpha)
def clip(x): #essential thing to avoid stack overflow, to clip weights and biases
    return np.clip(x,-0.1,0.1)

#initialize variables
epochs=2000 #training epochs
lr=1e-7 #learning rate
bs=1500 #batch size (training input)
mse_threshold=1e-4 #threshold, after which training stops
old_l=1 #a variable to control the change in the loss function and thus change learning rate
l=1 #loss function
t_bs=5 #test batch size
non1=100 #number of neurons 1st layer
non2=100
non3=100
non4=100
non5=100
non6=100
non7=1 #number of neurons 7th (output) layer
inputs=np.random.randint(-10, 10, size=(bs,2)) #input is a random matrix
targets=np.zeros((bs,1)) #another cool thing I created:
for i in range(bs):
    targets[i,0]=inputs[i,0]+inputs[i,1] #automatically sums all the random numbers generated earlier
#print (f"inputs\n{inputs}") #uncomment if yiu want to see what was generated as inputs and targets
#print (f"targets\n{targets}")
w1=np.random.randn(2,non1) #weights initialization
w2=np.random.randn(non1,non2) #the dimensions as variables!
w3=np.random.randn(non2,non3)
w4=np.random.randn(non3,non4)
w5=np.random.randn(non4,non5)
w6=np.random.randn(non5,non6)
w7=np.random.randn(non6,non7)
b1=np.zeros((1,non1)) #biases initialization
b2=np.zeros((1,non2))
b3=np.zeros((1,non3))
b4=np.zeros((1,non4))
b5=np.zeros((1,non5))
b6=np.zeros((1,non6))
b7=np.zeros((1,non7))
#training
for epoch in range(epochs):
    if epoch<epochs/(epochs/10) and l>0.3: #when learning starts, the loss function can be HUGE, so to avoid HUGE gradients of weights and biases I manually limited the leearning rate for the beginning of the program
        lr=1e-6
    #forward pass
    z1=inputs@w1+b1 #z1 are unactivated outputs, w1 is weight, b1 is bias
    a1=leaky_relu(z1) #a1 is activated output (or not activated, like in the 7th layer)
    z2=a1@w2+b2
    a2=leaky_relu(z2)
    z3=a2@w3+b3
    a3=leaky_relu(z3)
    z4=a3@w4+b4
    a4=leaky_relu(z4)
    z5=a4@w5+b5
    a5=leaky_relu(z5)
    z6=a5@w6+b6
    a6=leaky_relu(z6)
    z7=a6@w7+b7
    a7=z7
    #backpropagation
    l=np.mean(np.square(a7-targets), dtype=np.float64) #loss is mean squared error, np.float64 is to increase the precision of calcualtion (ChatGPT adviced me to do it, I think it helps)
    z7grad=2*(a7-targets) #if a7 (prediction) is too high, gradient is high, so we subtract gradient from weight
    z6grad=z7grad@w7.T*leaky_relu_derivative(z6) #chain rule of calculus
    z5grad=z6grad@w6.T*leaky_relu_derivative(z5)
    z4grad=z5grad@w5.T*leaky_relu_derivative(z4)
    z3grad=z4grad@w4.T*leaky_relu_derivative(z3)
    z2grad=z3grad@w3.T*leaky_relu_derivative(z2)
    z1grad=z2grad@w2.T*leaky_relu_derivative(z1)
    w7grad=a6.T@z7grad
    w6grad=a5.T@z6grad
    w5grad=a4.T@z5grad
    w4grad=a3.T@z4grad
    w3grad=a2.T@z3grad
    w2grad=a1.T@z2grad
    w1grad=inputs.T@z1grad
    b7grad=np.sum(z7grad, axis=0, keepdims=True)
    b6grad=np.sum(z6grad, axis=0, keepdims=True)
    b5grad=np.sum(z5grad, axis=0, keepdims=True)
    b4grad=np.sum(z4grad, axis=0, keepdims=True)
    b3grad=np.sum(z3grad, axis=0, keepdims=True)
    b2grad=np.sum(z2grad, axis=0, keepdims=True)
    b1grad=np.sum(z1grad, axis=0, keepdims=True)
    #weights and biases updates
    w7-=w7grad*lr
    w6-=w6grad*lr
    w5-=w5grad*lr
    w4-=w4grad*lr
    w3-=w3grad*lr
    w2-=w2grad*lr
    w1-=w1grad*lr
    b7-=b7grad*lr
    b6-=b6grad*lr
    b5-=b5grad*lr
    b4-=b4grad*lr
    b3-=b3grad*lr
    b2-=b2grad*lr
    b1-=b1grad*lr
    #clipping to avoid stack overflow
    w7=clip(w7) #helps with stack overflow and doesn't cause a significant drop in performance
    w6=clip(w6)
    w5=clip(w5)
    w4=clip(w4)
    w3=clip(w3)
    w2=clip(w2)
    w1=clip(w1)
    b7=clip(b7)
    b6=clip(b6)
    b5=clip(b5)
    b4=clip(b4)
    b3=clip(b3)
    b2=clip(b2)
    b1=clip(b1)
    #showing mean squared error, break condition
    if epoch%(epochs/10)==0 or epoch==epochs-1:
        print (f"epoch {epoch}, MSE {l}") #print epoch and loss function
    if l<mse_threshold:
        print (f"epoch {epoch}, MSE {l}, BREAK") #print epoch and loss, when training is stopped earlier
        break
    #I removed dynamic learning rate, because it doesn't seem to do much (I couldn't make it good enough)
    if epoch<10: #print epoch and mean squared error for first 10 epochs (though it prints epoch 0 for the 2nd time)
        print (f"epoch {epoch}, MSE {l}")
        
    #inference
test_inputs=np.random.randint(-10,10,size=(t_bs,2)) #the same method to create test inputs and targets, as for the training data
test_targets=np.zeros((t_bs,1))
for i in range(t_bs):
    test_targets[i,0]=inputs[i,0]+inputs[i,1]
tz1=test_inputs@w1+b1 #forward pass without packpropagation (just applying weights)
ta1=leaky_relu(tz1)
tz2=ta1@w2+b2
ta2=leaky_relu(tz2)
tz3=ta2@w3+b3
ta3=leaky_relu(tz3)
tz4=ta3@w4+b4
ta4=leaky_relu(tz4)
tz5=ta4@w5+b5
ta5=leaky_relu(tz5)
tz6=ta5@w6+b6
ta6=leaky_relu(tz6)
tz7=ta6@w7+b7
ta7=tz7
for i in range (t_bs): #for i in the test batch size (i in the number of rows)
    print (f"({test_inputs[i,0]})+({test_inputs[i,1]})=({ta7[i,0]:.1f})") #only 1 figure after the dot 

# OUTPUT:
# epoch 0, MSE 180943570586.75974
# epoch 0, MSE 180943570586.75974
# epoch 1, MSE 230.9164955132931
# epoch 2, MSE 125.1650633195648
# epoch 3, MSE 90.46125729996665
# epoch 4, MSE 70.5029820492783
# epoch 5, MSE 64.6341023366769
# epoch 6, MSE 59.98401001993726
# epoch 7, MSE 51.405395518459635
# epoch 8, MSE 36.27316082849468
# epoch 9, MSE 14.02297029621542
# epoch 200, MSE 0.2686835153184475
# epoch 400, MSE 0.10821340534350035
# epoch 600, MSE 0.05850103905039577
# epoch 800, MSE 0.036395450102987274
# epoch 1000, MSE 0.02892381154449545
# epoch 1200, MSE 0.02158771864394958
# epoch 1400, MSE 0.017542450141767883
# epoch 1600, MSE 0.014106738672535861
# epoch 1800, MSE 0.0124740690974283
# epoch 1999, MSE 0.010721822721369118
# (4)+(-5)=(-1.0)
# (-8)+(9)=(1.1)
# (-1)+(-6)=(-6.9)
# (-3)+(-6)=(-8.9)
# (1)+(-2)=(-1.0)
