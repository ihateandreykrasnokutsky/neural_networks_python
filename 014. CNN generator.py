#make a CNN that generates a picture
#deepseek said that I can use the same weights for covolution and deconvolution (it's my idea, isn't it cool?), and that it's a solid approach
#reached the tanh(x) line, understanding well everything
 #adding the padding AFTER the cycle, because if we do it IN the cycle, then the kernel can't be applied without going out of edges of the output. We could cut the kernel inside, but damn that's mouthful/cumbersome (in terms of the amount of code) approach
#It works! And I understand it. The next step is the actual image generation, using this structure. It seems like nothing difficult or unknown (yet), basically forward pass and backprop through FC and transposed concoluton.
#OUTPUT: Output image's shape:  (3, 32, 32)

import numpy as np

def relu(x):
    return np.maximum(0,x)

def tanh(x):
    return np.tanh(x)

def conv_transpose2d(x,weight,stride=2,padding=1):
    C_in,H,W=x.shape
    C_in_w,C_out,kH,kW=weight.shape
    assert C_in==C_in_w, "Input channels must match weight channels"

    #compute output size
    H_out=kH+(H-1)*stride
    W_out=kW+(W-1)*stride
    out=np.zeros((C_out,H_out,W_out))
    for c_in in range(C_in):
        for c_out in range(C_out):
            for i in range(H):
                for j in range(W):
                    out[c_out,i*stride:i*stride+kH,j*stride:j*stride+kW]+=x[c_in,i,j]*weight[c_in,c_out,:,:]
    if padding>0:
        out=out[:,padding:-padding,padding:-padding]
    return out
class CNNGenerator:
    def __init__(self,latent_dim=100):
        self.latent_dim=latent_dim
        #init weights
        self.fc_weight=np.random.randn(latent_dim,4*4*256)*0.02
        #transposed convolution weights (C_in, C_out, kH, kW)
        self.ct1_weight=np.random.randn(256,128,4,4)*0.02
        self.ct2_weight=np.random.randn(128,64,4,4)*0.02
        self.ct3_weight=np.random.randn(64,3,4,4)*0.02
    def forward(self,z):
        x=z@self.fc_weight #latent_dim=>(4*4*256)
        x=x.reshape(256,4,4) #to 256x4x4
        x=conv_transpose2d(x,self.ct1_weight) #to 128x8x8
        x=relu(x)
        x=conv_transpose2d(x,self.ct2_weight) #to 64x16x16
        x=relu(x)
        x=conv_transpose2d(x,self.ct3_weight) #to 3x32x32
        x=tanh(x) #this gives out pixels in range [-1,1], so you need to transfrom it to [0,1] or [0,255], if you want a real image
        return x
    
#example usage
latent_dim=100
z=np.random.randn(latent_dim)
generator=CNNGenerator(latent_dim)
fake_image=generator.forward(z)

print ("Output image's shape: ",fake_image.shape)
    





    