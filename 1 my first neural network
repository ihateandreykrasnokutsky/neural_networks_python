#but the question is, how to add more weights and biases, because my neural network basically has 1 layer...
#well, at least it has a non-linear activation function, kinda redundant for such task

import numpy as np

inputs = np.array([[0.1],[0.2],[0.13],[0.14],[0.15],[0.6],[0.17],[0.18],[0.19],[0.111],[0.112],[0.114],[0.115],[0.1111]])
targets=inputs*3#changing from np.dot(inputs,3) to this helped to solve dimnensional problems in the program, idk what exactly happened though

weight=np.random.rand(1,1)#While working on the program I realized that I don't need to initialize weights in the same number of them as the number of inputs, because when I'll multiply them later (to get the linear output, with the formula z=w*x+b), I'll have the matrix of the linear output of the needed size (equuivalent to the size of the input).
bias=np.random.rand(1,1)

def relu(x):
    return np.maximum(0,x)
def relu_derivative(x):
    return np.where(x>0,1,0)

learning_rate=0.1

for epoch in range(1000):
    linear_output=np.dot(inputs,weight)+bias
    activated_output=relu(linear_output)
    error=targets-linear_output
    gradient=error*relu_derivative(activated_output)
    weight+=np.dot(inputs.T,gradient)*learning_rate#does weight variable changes it's dimensions as the result (from 1,1 to 14,1, for each input)?
    bias+=np.sum(gradient)*learning_rate#the same question for bias
    mse=np.mean(error**2)
    if (epoch%100==0): print (f"epoch {epoch}: MSE={mse}")
    if (mse<10**-10):
        print ("\nBREAK\n")
        break


test_inputs=[[1],[3],[2],[4],[2],[1],[3],[2],[4],[2],]
test_output=np.dot(test_inputs,weight)+bias
print(f"test input is:\n{test_inputs}\ntest output is:\n{test_output}")


