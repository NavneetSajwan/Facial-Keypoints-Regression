## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
# import torch.nn.init as I


def conv(n_inputs, n_filters, kernel_size=3, stride=1, bias=False) -> torch.nn.Conv2d:
    """Creates a convolution layer for `XResNet`."""
    return nn.Conv2d(n_inputs, n_filters,
                     kernel_size=kernel_size, stride=stride,
                     padding=kernel_size//2, bias=bias) # if S =1, and ks = 3, 

def conv_layer(n_inputs: int, n_filters: int,
               kernel_size: int = 3, stride=1,
               zero_batch_norm: bool = False, use_activation: bool = True,
               activation: torch.nn.Module = nn.ReLU(inplace=True)) -> torch.nn.Sequential:
    """Creates a convolution block for `XResNet`."""
    batch_norm = nn.BatchNorm2d(n_filters)
    # initializer batch normalization to 0 if its the final conv layer
    nn.init.constant_(batch_norm.weight, 0. if zero_batch_norm else 1.)
    layers = [conv(n_inputs, n_filters, kernel_size, stride=stride), batch_norm]
    if use_activation: layers.append(activation)
    return nn.Sequential(*layers)


class XResNetBlock(nn.Module):
    """Creates the standard `XResNet` block."""
    def __init__(self, expansion: int, n_inputs: int, n_hidden: int, stride: int = 1,
                 activation: torch.nn.Module = nn.ReLU(inplace=True)):
        super().__init__()

        n_inputs = n_inputs * expansion
        n_filters = n_hidden * expansion

        # convolution path
        if expansion == 1:
            layers = [conv_layer(n_inputs, n_hidden, 3, stride=stride), # In this, since the s=1, op size is same as ip size
                      conv_layer(n_hidden, n_filters, 3, zero_batch_norm=True, use_activation=False)]
        else:
            layers = [conv_layer(n_inputs, n_hidden, 1),
                      conv_layer(n_hidden, n_hidden, 3, stride=stride),
                      conv_layer(n_hidden, n_filters, 1, zero_batch_norm=True, use_activation=False)]

        self.convs = nn.Sequential(*layers)

        # identity path
        if n_inputs == n_filters:
            self.id_conv = nn.Identity()
        else:
            self.id_conv = conv_layer(n_inputs, n_filters, kernel_size=1, use_activation=False)
        if stride == 1:
            self.pool = nn.Identity()
        else:
            self.pool = nn.AvgPool2d(2, ceil_mode=True)

        self.activation = activation

    def forward(self, x):
        return self.activation(self.convs(x) + self.id_conv(self.pool(x)))



class XResNet(nn.Sequential):
    @classmethod
    def create(cls, expansion, layers, c_in=3, c_out=1000):
        # create the stem of the network
        n_filters = [c_in, (c_in+1)*8, 64, 64]
        stem = [conv_layer(n_filters[i], n_filters[i+1], stride=2 if i==0 else 1)
                for i in range(3)]

        # create `XResNet` blocks
        n_filters = [64//expansion, 64, 128, 256, 512]

        res_layers = [cls._make_layer(expansion, n_filters[i], n_filters[i+1],
                                      n_blocks=l, stride=1 if i==0 else 2)
                      for i, l in enumerate(layers)]

        # putting it all together
        x_res_net = cls(*stem, nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                        *res_layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                        nn.Linear(n_filters[-1]*expansion, c_out)
                       )

        cls._init_module(x_res_net)
        return x_res_net

    @staticmethod
    def _make_layer(expansion, n_inputs, n_filters, n_blocks, stride):
        return nn.Sequential(
            *[XResNetBlock(expansion, n_inputs if i==0 else n_filters, n_filters, stride if i==0 else 1)
              for i in range(n_blocks)])

    @staticmethod
    def _init_module(module):
        if getattr(module, 'bias', None) is not None:
            nn.init.constant_(module.bias, 0)
        if isinstance(module, (nn.Conv2d,nn.Linear)):
            nn.init.kaiming_normal_(module.weight)
        # initialize recursively
        for l in module.children():
            XResNet._init_module(l)



# ## TODO: define the convolutional neural network architecture

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# # can use the below import should you choose to initialize the weights of your Net
# import torch.nn.init as I

# def conv(ni, nf):
# 	return nn.Conv2d(ni, nf, kernel_size = 3, stride = 2, padding = 1)

# def conv_layer(ni, nf):
# 	return nn.Sequential(conv(ni,nf), nn.BatchNorm2d(nf, eps=1e-05, momentum=0.1), nn.ReLU())

# # def convres(ni, nf):
	

# class Net(nn.Module):

# 	def __init__(self):
# 		super(Net, self).__init__()

# 		## TODO: Define all the layers of this CNN, the only requirements are:
# 		## 1. This network takes in a square (same width and height), grayscale image as input
# 		## 2. It ends with a linear layer that represents the keypoints
# 		## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs

# 		# As an example, you've been given a convolutional layer, which you may (but don't have to) change:
# 		# 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
# 		self.layer1 = nn.Sequential(conv_layer(1,8),# 128
# 					    conv_layer(8,16),# 64
# 					    nn.MaxPool2d(2,2),#32
# 					    conv_layer(16,32), #16
# 					    conv_layer(32,64), #8
# 					    nn.MaxPool2d(2,2),#4
# 					    conv_layer(64,128),#2
# 					    conv_layer(128,136),
# 					    nn.Tanh())#1

# # 		self.layer2 = nn.Sequential(conv_layer(3,8),# 128
# # 						# conv_layer(4,8),# 128
# # 						conv_layer(8,16),# 64
# # 						nn.MaxPool2d(2,2),#32
# # 						conv_layer(16,32), #16
# # 						# nn.MaxPool2d(2,2)#16
# # 						conv_layer(32,64), #8
# # 						nn.MaxPool2d(2,2),
# # 						nn.ReLU())#4

# 		self.layer3 = nn.Sequential(nn.Linear(1024, 136), nn.Tanh())
# 		## Note that among the layers to add, consider including:
# 		# maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        


# 	def forward(self, x):
#         ## TODO: Define the feedforward behavior of this model
#         ## x is the input image and, as an example, here you may choose to include a pool/conv step:
#         ## x = self.pool(F.relu(self.conv1(x)))
        
# #         x = 1.2*self.layer1(x)
# 		x = self.layer2(x)
# 		x = 1.2 *self.layer3(x.view(-1, 1024))

# 		# a modified x, having gone through all the layers of your model, should be returned
# 		return x
