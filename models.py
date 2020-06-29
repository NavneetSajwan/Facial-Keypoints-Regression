## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

def conv(ni, nf):
	return nn.Conv2d(ni, nf, kernel_size = 3, stride = 2, padding = 1)

def conv_layer(ni, nf):
	return nn.Sequential(conv(ni,nf), nn.BatchNorm2d(nf), nn.ReLU())

def convres(ni, nf):
	

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.layer1 = nn.Sequential(conv_layer(3,8),# 128
				# conv_layer(4,8),# 128
				conv_layer(8,16),# 64
				nn.MaxPool2d(2,2),#32
				conv_layer(16,32), #16
				# nn.MaxPool2d(2,2)#16
				conv_layer(32,64), #8
				nn.MaxPool2d(2,2),#4
				conv_layer(64,128),#2
				conv_layer(128,136),
				nn.Tanh())#1

        self.layer2 = nn.Sequential(conv_layer(3,8),# 128
				# conv_layer(4,8),# 128
				conv_layer(8,16),# 64
				nn.MaxPool2d(2,2),#32
				conv_layer(16,32), #16
				# nn.MaxPool2d(2,2)#16
				conv_layer(32,64), #8
				nn.MaxPool2d(2,2)
				nn.ReLU())#4
	
	self.layer3 = nn.Sequential(nn.Linear(1024, 136), 1.2* nn.Tanh())
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        


    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
#         x = 1.2*self.layer1(x)
	x = self.layer2(x)
	x = self.layer3(x.view(-1, 1024))
	
        # a modified x, having gone through all the layers of your model, should be returned
        return x
