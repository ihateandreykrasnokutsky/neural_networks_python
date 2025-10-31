#make a CNN that generates a picture
#deepseek said that I can use the same weights for covolution and deconvolution (it's my idea, isn't it cool?), and that it's a solid approach

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
    H_out=kH+(H-1)*stride-2*padding
    W_out=kW+(W-1)*stride-2*padding
    out=np.zeros((C_out,H_out,W_out))

    


    