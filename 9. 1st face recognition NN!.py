from PIL import Image #the program that determines if the picture on the image is happy or sad face
import numpy as np
np.set_printoptions(threshold=np.inf) #disable the limitations to print large arays in the terminal

cmp_size=50 #compressed image's size
hid_nrn=10 #number of hidden neurons
mse_threshold=1e-5
lr=0.01
epochs=1000

def image_to_matrix(image_path, size=(cmp_size,cmp_size)):
    img=Image.open(image_path).convert('L')
    img=img.resize(size)
    matrix=np.array(img) #0-255, black-white
    matrix=(matrix<128).astype(int) #<128 is 1 (black), >=128 is 0 (white)
    return matrix.flatten()
def leaky_relu(x, alpha=0.05):
    return np.where(x>0, x, x*alpha)
def leaky_relu_derivative(x, alpha=0.05):
    return np.where(x>0, 1, alpha)
def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))

happy_faces=[image_to_matrix(r"D:\Pictures\machine_learning\smiles_data\xh"+str(i)+".png") for i in range(40)]
sad_faces=[image_to_matrix(r"D:\Pictures\machine_learning\smiles_data\xs"+str(i)+".png") for i in range(40)]
x=np.vstack(happy_faces+sad_faces)#(80,2500)
y=np.array([[1]]*40+[[0]]*40)#(80,1)
w1=np.random.randn(cmp_size*cmp_size,hid_nrn)*np.sqrt(2.0/(cmp_size*cmp_size))
b1=np.zeros((1,hid_nrn))
w2=np.random.randn(hid_nrn,hid_nrn)*np.sqrt(2.0/hid_nrn)
b2=np.zeros((1,hid_nrn))
w3=np.random.randn(hid_nrn,hid_nrn)*np.sqrt(2.0/hid_nrn)
b3=np.zeros((1,hid_nrn))
w4=np.random.randn(hid_nrn,1)*np.sqrt(2.0/hid_nrn)
b4=np.zeros((1,1))

for epoch in range (epochs):
    #forward pass
    z1=x@w1+b1
    a1=leaky_relu(z1)
    z2=a1@w2+b2
    a2=leaky_relu(z2)
    z3=a2@w3+b3
    a3=leaky_relu(z3)
    z4=a3@w4+b4
    a4=sigmoid(z4)
    #backpropagation
    l=np.mean((a4-y)**2)
    z4grad=2*(a4-y)*sigmoid_derivative(z4)
    z3grad=z4grad@w4.T*leaky_relu_derivative(z3)
    z2grad=z3grad@w3.T*leaky_relu_derivative(z2)
    z1grad=z2grad@w2.T*leaky_relu_derivative(z1)
    w4grad=a3.T@z4grad
    w3grad=a2.T@z3grad
    w2grad=a1.T@z2grad
    w1grad=x.T@z1grad
    b4grad=np.sum(z4grad, axis=0, keepdims=True)
    b3grad=np.sum(z3grad, axis=0, keepdims=True)
    b2grad=np.sum(z2grad, axis=0, keepdims=True)
    b1grad=np.sum(z1grad, axis=0, keepdims=True)
    #updates
    w1-=w1grad*lr
    w2-=w2grad*lr
    w3-=w3grad*lr
    w4-=w4grad*lr
    b1-=b1grad*lr
    b2-=b2grad*lr
    b3-=b3grad*lr
    b4-=b4grad*lr
    #showing MSE
    if epoch%(epochs/10)==0 or epoch==epochs-1:
        print (f"epoch {epoch}, MSE {l}")
    if l<mse_threshold:
        print (f"epoch {epoch}, MSE {l}, BREAK")
        break

#inference
test_happy_faces=[image_to_matrix(r"D:\Pictures\machine_learning\smiles_data\txh"+str(i)+".png") for i in range(10)]
test_sad_faces=[image_to_matrix(r"D:\Pictures\machine_learning\smiles_data\txs"+str(i)+".png") for i in range(10)]
tx=np.vstack(test_happy_faces+test_sad_faces)
print (f"\nTest matrix is 10 happy faces + 10 sad faces.")
#test forward pass
tz1=tx@w1+b1
ta1=leaky_relu(tz1)
tz2=ta1@w2+b2
ta2=leaky_relu(tz2)
tz3=ta2@w3+b3
ta3=leaky_relu(tz3)
tz4=ta3@w4+b4
ta4=sigmoid(tz4)
ta4=np.where(ta4>0.5,1,0)
print(f"predictions about faces are:\n{ta4}\nmeaning:")
text_ta4=np.where(ta4==1,"happy_face", "sad face")
print (text_ta4)

# for this program to learn and work, I made 40 pictures of happy faces (just simplistic smiles, I'll upload them too)
# and 40 sad faces, plus 10 happy and 10 sad for the inference.
# The program works very stable! Though it stubbornly considers one sad face happy (txs1.png). Most likely,
# it just was drawn not so clearly with a distortion of the smile,
# so the neural network was confused
# but I'm  happy that the network I wrote can recognize real pictures, and the network
# required for this job is pretty simple!
