import torch
import torchvision.models as models
import torchvision.transforms as trans
import skimage.io as skio
import skimage.transform as sktr
import skimage.filters as skfltr
import skimage.util as skutil
import skimage.color as skclr
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

vgg = models.vgg19()
