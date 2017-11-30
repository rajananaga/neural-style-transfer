import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as trans
import skimage.io as skio
import skimage.util as skutil
import matplotlib.pyplot as plt
import numpy as np

IM_PATH = 'input/'
F_EXT = 'JPG'
CONTENT_IMAGE = IM_PATH + 'content.jpg'
STYLE_IMAGE = IM_PATH + 'style.jpg'
IM_SIZE = 512

def toTorch(im):
    im = trans.Resize(IM_SIZE)(content)
    im = Variable(trans.ToTensor()(content))
    # VGG network throws error if the shape doesn't have a 1 in front (1 x 512 x 512)
    im = im.unsqueeze(0)
    return im

def load_images():
    content = skio.imread(CONTENT_IMAGE)
    style = skio.imread(STYLE_IMAGE)
    content = toTorch(content)
    style = toTorch(style)
    assert style.shape == content.shape, "Image shapes are not equal"
    return content, style

def initialize_target_image(shape):
    im = np.zeros(shape)
    return skutil.random_noise(im)

# print(models.vgg19().__dict__)

class VGG(nn.Module):
    def __init__(self):
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
content_results = vgg.forward(content)
style_results = vgg.forward(style)
