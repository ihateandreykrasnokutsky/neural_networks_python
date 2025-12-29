# ===============================================================
#  TRANPOSED CNN IMAGE GENERATOR (SUPERVISED, NOT GAN)
#  Input: random noise vector z
#  Output: generated image
#  Training signal: real image (supervised)
#
#  Loss: Mean Squared Error (MSE)
#  Optimizer: simple gradient descent
#
#  A VERY SMALL educational example.
# ===============================================================

import numpy as np
from skimage import io, transform
from datetime import datetime

# -----------------------------
# Hyperparameters
# -----------------------------
epochs = 10000
latent_dim = 10
learning_rate = 1
target_size = (64, 64)  # final generated image size

# -----------------------------
# Utility functions
# -----------------------------

# -----------------------------
# Save generated image
# -----------------------------
def save_image(tensor_img, path):
    """
    tensor_img: numpy array of shape (3, H, W), values in [-1, 1]
    Saves to 'path' as a JPEG.
    """
    img = tensor_img.copy()
    img = (img + 1) / 2         # [-1,1] -> [0,1]
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    img = img.transpose(1, 2, 0)  # CxHxW -> HxWxC

    io.imsave(path, img)

def load_image(path, target_size):
    """
    Loads an image, resizes it to target_size,
    normalizes it to [-1, 1] so it matches tanh output.
    """
    img = io.imread(path).astype(np.float32) / 255.0
    img = transform.resize(img, target_size, anti_aliasing=True)
    img = img.transpose(2, 0, 1)  # convert to C x H x W
    img = img * 2 - 1            # move to [-1, 1]
    return img

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    # derivative of ReLU: 1 where x>0 else 0
    return (x > 0).astype(np.float32)

def tanh(x):
    return np.tanh(x)

def tanh_grad(x):
    # derivative of tanh: 1 - tanh(x)^2
    return 1 - np.tanh(x)**2


# ===============================================================
#  TRANSPOSED CONVOLUTION (FORWARD + BACKWARD)
# ===============================================================

def conv_transpose2d_forward(x, w, stride=2, padding=1):
    """
    x: input feature map  (C_in, H, W)
    w: weights            (C_in, C_out, kH, kW)
    returns: output feature map (C_out, H_out, W_out)

    This does NOT implement real padding; the output is cropped
    at the end to simulate "removing" padding.
    """
    C_in, H, W = x.shape
    C_in_w, C_out, kH, kW = w.shape
    assert C_in == C_in_w

    H_out = kH + (H - 1)*stride
    W_out = kW + (W - 1)*stride

    out = np.zeros((C_out, H_out, W_out), dtype=np.float32)

    # Deconvolution (transposed conv)
    for cin in range(C_in):
        for cout in range(C_out):
            for i in range(H):
                for j in range(W):
                    out[cout,
                        i*stride : i*stride + kH,
                        j*stride : j*stride + kW] += x[cin, i, j] * w[cin, cout]

    # Remove "padding" by simple cropping
    if padding > 0:
        out = out[:, padding:-padding, padding:-padding]

    return out


def conv_transpose2d_backward(x, w, grad_out, stride=2, padding=1):
    """
    Computes gradients for:
      dL/dx  and  dL/dw

    grad_out: gradient from next layer (C_out, H_out, W_out)
    """

    # First reconstruct the pre-cropped grad_out
    C_in, H, W = x.shape
    C_in_w, C_out, kH, kW = w.shape

    H_out_full = kH + (H - 1)*stride
    W_out_full = kW + (W - 1)*stride

    # Add padding back
    if padding > 0:
        grad_full = np.zeros((C_out, H_out_full, W_out_full), dtype=np.float32)
        grad_full[:, padding:-padding, padding:-padding] = grad_out
    else:
        grad_full = grad_out

    # Initialize gradients
    grad_x = np.zeros_like(x)
    grad_w = np.zeros_like(w)

    # Compute gradients
    for cin in range(C_in):
        for cout in range(C_out):
            for i in range(H):
                for j in range(W):
                    # dW
                    grad_w[cin, cout] += x[cin, i, j] * grad_full[cout,
                                                                   i*stride : i*stride+kH,
                                                                   j*stride : j*stride+kW]
                    # dX
                    grad_x[cin, i, j] += np.sum(w[cin, cout] *
                                                grad_full[cout,
                                                          i*stride : i*stride+kH,
                                                          j*stride : j*stride+kW])

    return grad_x, grad_w


# ===============================================================
#  GENERATOR CLASS
# ===============================================================

class CNNGenerator:
    def __init__(self):
        """
        Initialize weights.
        We create:
          - FC weight: latent_dim -> 256*4*4
          - 4 transpose-convs:
                256 -> 128 channels (layers)
                128 -> 64
                 64 -> 32  
                 64 -> 3 (RGB)
        """
        self.latent_dim = latent_dim
        self.fc_weight = np.random.randn(latent_dim, 256*4*4) * 0.02

        self.ct1_weight = np.random.randn(256, 128, 4, 4) * 0.02
        self.ct2_weight = np.random.randn(128, 64, 4, 4) * 0.02
        self.ct3_weight = np.random.randn(64, 32, 4, 4) * 0.02
        self.ct4_weight = np.random.randn(32, 3, 4, 4) * 0.02

    # -----------------------------
    # FORWARD PASS
    # -----------------------------
    def forward(self, z):
        """
        z is latent noise of shape (latent_dim,)
        """

        # FC Layer
        self.z = z
        self.fc_out = z @ self.fc_weight          # shape (256*4*4)
        self.x1 = self.fc_out.reshape(256, 4, 4)  # shape (256, 4, 4)

        # First transposed conv
        self.y1 = conv_transpose2d_forward(self.x1, self.ct1_weight)
        self.y1_relu = relu(self.y1)

        # Second
        self.y2 = conv_transpose2d_forward(self.y1_relu, self.ct2_weight)
        self.y2_relu = relu(self.y2)

        # Third
        self.y3 = conv_transpose2d_forward(self.y2_relu, self.ct3_weight)
        self.y3_relu=relu(self.y3)

        #Fourth
        self.y4=conv_transpose2d_forward(self.y3_relu, self.ct4_weight)
        self.out = tanh(self.y4)

        return self.out

    # -----------------------------
    # BACKWARD PASS (GRADIENTS)
    # -----------------------------
    def backward(self, grad_out):
        """
        grad_out is dLoss/dOutput (same shape as generated image)
        """

        # Tanh layer
        grad_y4 = grad_out * tanh_grad(self.y4)

        # ConvTranspose4
        grad_y3_relu, grad_ct4 = conv_transpose2d_backward(
            self.y3_relu, self.ct4_weight, grad_y4
        )

        # ReLU3
        grad_y3 = grad_y3_relu * relu_grad(self.y3)

        # ConvTranspose3
        grad_y2_relu, grad_ct3 = conv_transpose2d_backward(
            self.y2_relu, self.ct3_weight, grad_y3
        )

        # ReLU2
        grad_y2 = grad_y2_relu * relu_grad(self.y2)

        # ConvTranspose2
        grad_y1_relu, grad_ct2 = conv_transpose2d_backward(
            self.y1_relu, self.ct2_weight, grad_y2
        )

        # ReLU1
        grad_y1 = grad_y1_relu * relu_grad(self.y1)

        # ConvTranspose1
        grad_x1, grad_ct1 = conv_transpose2d_backward(
            self.x1, self.ct1_weight, grad_y1
        )

        # FC layer gradient
        grad_fc = grad_x1.reshape(256*4*4)
        grad_fc_w = np.outer(self.z, grad_fc)

        return grad_ct1, grad_ct2, grad_ct3, grad_ct4, grad_fc_w

    # -----------------------------
    # Update weights
    # -----------------------------
    def step(self, grads):
        grad_ct1, grad_ct2, grad_ct3, grad_ct4, grad_fc = grads

        # Gradient descent
        self.ct1_weight -= learning_rate * grad_ct1
        self.ct2_weight -= learning_rate * grad_ct2
        self.ct3_weight -= learning_rate * grad_ct3
        self.ct4_weight -= learning_rate * grad_ct4
        self.fc_weight  -= learning_rate * grad_fc


# ===============================================================
#  TRAINING LOOP
# ===============================================================

target_img = load_image('./neural_networks_python/015-CNN-generator-data/cat_0.jpg', target_size)  # must be CxHxW

gen = CNNGenerator()

# random noise (I moved it outside of the "for" loop, because the randomly infinitely changing input messes up with the gradients a lot, making the task impossible)
z = np.random.randn(latent_dim).astype(np.float32)

for epoch in range(epochs):

    # forward pass
    fake = gen.forward(z)

    # compute MSE loss and gradient
    diff = fake - target_img
    loss = np.mean(diff**2)
    grad_out = (2 / diff.size) * diff

    # backward
    grads = gen.backward(grad_out)

    # update
    gen.step(grads)

    # print & save images
    if epoch % 1 == 0:
        print(f"Epoch {epoch}, Loss = {loss:.5f}, Learning Rate = {learning_rate}")

    # save every 5 epochs
    if epoch % 1 == 0:
        out_path = f'./neural_networks_python/015-CNN-generator-data/generated_epoch_{epoch}.jpg'
        save_image(fake, out_path)
        print(f"[{datetime.now().strftime('%H:%M:%S')}]: Saved image for epoch {epoch} -> {out_path}")

print("Training complete.")
print("Final output shape:", fake.shape)

# MY COMMENTS
# I added the correct path for images, changed the images to jpg format (png can save the 4th alpha channel that I don't need, jpg can't). The learning doesn't move the model anywhere during 136 epochs, 0.01 learning rate. Need to try 0.1 learning rate or higher, because the loss just dances around ~0.83566.
# Now the point is to make it produce 64x64 images. Given that I didn't write and didn't understand the program fully, it won't be easy. But it's good enough idea to try.
# It works, now I need it to use a few different images as labels, because CNNs are not for copying, but for understanding meaningful features
