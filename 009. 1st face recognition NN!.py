from PIL import Image #the program that determines if the picture on the image is happy or sad face
import numpy as np
np.set_printoptions(threshold=np.inf) #disable the limitations to print large arays in the terminal

cmp_size=50 #compressed image's size
hid_nrn=10 #number of hidden neurons
mse_threshold=1e-5 #the minimal mean squared error (loss function). The training loop stops after reaching it.
lr=0.01 #learning rate
epochs=1000 #the number if iterations for the  training

def image_to_matrix(image_path, size=(cmp_size,cmp_size)): #this function transforms the image (happy/sad face) to a matrix
    img=Image.open(image_path).convert('L') #removes all colors, except black and white
    img=img.resize(size) #reducing the size of the image
    matrix=np.array(img) #transforms the image into a matrix, 0 is black, 255 is white
    matrix=(matrix<128).astype(int) #if the pixel is <128, then it =1 (black), else white (to remove the gray colors from the picture and make it simpler)
    return matrix.flatten() #transforms the rectangular matrix into a flat matrix (to make it easier for the NN to work with it)
def leaky_relu(x, alpha=0.05): #activation function for hidden layers
    return np.where(x>0, x, x*alpha)
def leaky_relu_derivative(x, alpha=0.05):#derivative of the activation function for the hidden layers
    return np.where(x>0, 1, alpha)
def sigmoid(x): #sigmoid for the output layer, because the task is classification
    return 1/(1+np.exp(-x))
def sigmoid_derivative(x): #the derivative of sigmoid
    return sigmoid(x)*(1-sigmoid(x))

happy_faces=[image_to_matrix(r"D:\Pictures\machine_learning\smiles_data\xh"+str(i)+".png") for i in range(40)] #input data for training. Imports pictures of happy faces and transforms them into matrices
sad_faces=[image_to_matrix(r"D:\Pictures\machine_learning\smiles_data\xs"+str(i)+".png") for i in range(40)] #same, but for sad faces
x=np.vstack(happy_faces+sad_faces)#(80,2500) #stacks all happy and sad faces' matrices on top of each other
y=np.array([[1]]*40+[[0]]*40)#(80,1) #the target (correct answers), 1 is happy face, 0 is sad face
w1=np.random.randn(cmp_size*cmp_size,hid_nrn)*np.sqrt(2.0/(cmp_size*cmp_size)) #weights initialization
b1=np.zeros((1,hid_nrn)) #bias intitialization
w2=np.random.randn(hid_nrn,hid_nrn)*np.sqrt(2.0/hid_nrn)
b2=np.zeros((1,hid_nrn))
w3=np.random.randn(hid_nrn,hid_nrn)*np.sqrt(2.0/hid_nrn)
b3=np.zeros((1,hid_nrn))
w4=np.random.randn(hid_nrn,1)*np.sqrt(2.0/hid_nrn) #output layer weights
b4=np.zeros((1,1)) #output layer bias

for epoch in range (epochs):
    #forward pass
    z1=x@w1+b1 #z is unactivated output
    a1=leaky_relu(z1) #a is activated output
    z2=a1@w2+b2
    a2=leaky_relu(z2)
    z3=a2@w3+b3
    a3=leaky_relu(z3)
    z4=a3@w4+b4
    a4=sigmoid(z4)
    #backpropagation
    l=np.mean((a4-y)**2) #loss function
    z4grad=2*(a4-y)*sigmoid_derivative(z4) #gradient of unactivated output
    z3grad=z4grad@w4.T*leaky_relu_derivative(z3)
    z2grad=z3grad@w3.T*leaky_relu_derivative(z2)
    z1grad=z2grad@w2.T*leaky_relu_derivative(z1)
    w4grad=a3.T@z4grad #gradient of the weight
    w3grad=a2.T@z3grad
    w2grad=a1.T@z2grad
    w1grad=x.T@z1grad
    b4grad=np.sum(z4grad, axis=0, keepdims=True) #gradient of the bias
    b3grad=np.sum(z3grad, axis=0, keepdims=True)
    b2grad=np.sum(z2grad, axis=0, keepdims=True)
    b1grad=np.sum(z1grad, axis=0, keepdims=True)
    #weight and bias updates
    w1-=w1grad*lr
    w2-=w2grad*lr
    w3-=w3grad*lr
    w4-=w4grad*lr
    b1-=b1grad*lr
    b2-=b2grad*lr
    b3-=b3grad*lr
    b4-=b4grad*lr
    #showing MSE
    if epoch%(epochs/10)==0 or epoch==epochs-1: #shows MSE every 1/10 of the total number of epochs, and near the end of learning
        print (f"epoch {epoch}, MSE {l}")
    if l<mse_threshold: #stop the training loop, if the certain value of the loss function is achieved
        print (f"epoch {epoch}, MSE {l}, BREAK")
        break

#inference
test_happy_faces=[image_to_matrix(r"D:\Pictures\machine_learning\smiles_data\txh"+str(i)+".png") for i in range(10)] #import pictures of happy faces and transform them to matrices, as the test input
test_sad_faces=[image_to_matrix(r"D:\Pictures\machine_learning\smiles_data\txs"+str(i)+".png") for i in range(10)] #same for the sad faces
tx=np.vstack(test_happy_faces+test_sad_faces) #vertically stack happy and sad faces' matrices
print (f"\nTest matrix is 10 happy faces + 10 sad faces.")
#test forward pass (same as in the training, but with the test data)
tz1=tx@w1+b1
ta1=leaky_relu(tz1)
tz2=ta1@w2+b2
ta2=leaky_relu(tz2)
tz3=ta2@w3+b3
ta3=leaky_relu(tz3)
tz4=ta3@w4+b4
ta4=sigmoid(tz4) #the result is the value between 0 and 1
ta4=np.where(ta4>0.5,1,0) #if the result of the sigmoid function is >0.5, then we round the value to 1 (to output the exact decision of the neural network (sad/happy face), and not just numbers numbers/probabilities inbetween)
print(f"predictions about faces are:\n{ta4}\nmeaning:") #print predictions (the corrected output of the sigmoid)
text_ta4=np.where(ta4==1,"happy_face", "sad face") #if the corrected output is 1, then it's happy face; if 0, then a sad one
print (text_ta4) #printing the output as the matrix, consisting of "happy face" and "sad face" values

# for this program to learn and work, I made 40 pictures of happy faces (just simplistic smiles, I'll upload them too)
# and 40 sad faces, plus 10 happy and 10 sad for the inference.
# The program works very stable! Though it stubbornly considers one sad face happy (txs1.png). Most likely,
# it just was drawn not so clearly with a distortion of the smile,
# so the neural network was confused
# but I'm  happy that the network I wrote can recognize real pictures, and the network
# required for this job is pretty simple!
