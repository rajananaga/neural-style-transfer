import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as trans
import skimage.io as skio
import skimage.util as skutil
import skimage.transform as sktrans
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable

IM_PATH = 'input/'
F_EXT = 'JPG'
CONTENT_IMAGE = IM_PATH + 'content.jpg'
STYLE_IMAGE = IM_PATH + 'style.jpg'
IM_SIZE = 512

def toTorch(im):
    im = sktrans.resize(im, (IM_SIZE, IM_SIZE, 3))
    im = Variable(torch.from_numpy(im))
    # VGG network throws error if the shape doesn't have a 1 in front (1 x 512 x 512)
    im = im.unsqueeze(0)
    return im

def load_images():
    content = skio.imread(CONTENT_IMAGE)/255.
    style = skio.imread(STYLE_IMAGE)/255.
    content = toTorch(content)
    style = toTorch(style)
    assert style.data.size() == content.data.size(), "Image shapes are not equal"
    return content, style

def initialize_target_image(shape):
    im = np.zeros(shape)
    return skutil.random_noise(im)

def calculate_content_loss(content_layers, style_layers):
    differences = []
    for i in content_layers:
        assert i in style_layers, "Layer mismatch"
        content, style = content_layers[i], style_layers[i]
        differences.append(0.5*(content - style)**2)

def calculate_style_loss(content_layers, style_layers):




class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.vgg = models.vgg19(pretrained=True)

    def forward(self, x):
        conv_results = {}
        for i, layer in enumerate(self.vgg.features):
            x = layer(x)
            if type(layer) == torch.nn.modules.conv.Conv2d:
                conv_results[i] = x
        return conv_results


vgg = VGG()
content, style = load_images()
content_layers = vgg.forward(content)
style_layers = vgg.forward(style)
