import sys
import torch
import torchvision.utils as utils
import torchvision.models as models
import torch.nn as nn
from torch.optim import LBFGS
import torchvision.transforms as trans
import skimage.io as skio
import skimage.util as skutil
import skimage.transform as sktrans
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import pdb

DATASET = 'starry'
IM_PATH = 'input/'
OUT_PATH = 'output/'
F_EXT = 'jpg'
CONTENT_IMAGE = IM_PATH + 'content_' + DATASET + '.' + F_EXT
STYLE_IMAGE = IM_PATH + 'style_' + DATASET + '.' + F_EXT
IM_SIZE = 512
IMAGE_SHAPE = (IM_SIZE, IM_SIZE, 3)
USE_CUDA = False
STYLE_WEIGHT = 1000
CONTENT_WEIGHT = 1
N_ITER = 500
STYLE_LAYER_WEIGHTS = [0.2 for _ in range(5)]
TENSOR_TYPE = torch.FloatTensor
CLONE_STYLE = False
CLONE_CONTENT = False

class VGGActivations(nn.Module):
    def __init__(self):
        super(VGGActivations, self).__init__()
        self.vgg = models.vgg19(pretrained=True)

    def forward(self, x):
        conv_results = []
        for i, layer in enumerate(self.vgg.features):
            x = layer(x)
            # Style: Conv1_1(0), Conv2_1(5), Conv3_1(10), Conv4_1(19), Conv5_1(28)
            # Content: Conv4_2(21)
            if type(layer) == torch.nn.modules.conv.Conv2d and i in [0, 5, 10, 19, 21, 28]:
                conv_results.append(x)
        return conv_results


# def toTorch(im):
#     im = sktrans.resize(im, IMAGE_SHAPE, mode='constant')
#     im = Variable(trans.ToTensor()(im))
#     # VGG network throws error if the shape doesn't have a 1 in front (1 x 512 x 512)
#     im = im.unsqueeze(0)
#     return im.type(TENSOR_TYPE)

def toTorch(im):
    im = sktrans.resize(im, IMAGE_SHAPE, mode='constant')
    transform = trans.Compose([trans.ToTensor(),trans.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    im = Variable(transform(im))
    # VGG network throws error if the shape doesn't have a 1 in front (1 x 512 x 512)
    im = im.unsqueeze(0)
    return im.type(TENSOR_TYPE)

def load_images():
    content = skio.imread(CONTENT_IMAGE)/1.
    style = skio.imread(STYLE_IMAGE)/1.
    content = toTorch(content)
    style = toTorch(style)
    assert style.data.size() == content.data.size(), "Image shapes are not equal"
    return content, style

def initialize_target_image():
    im = np.random.uniform(0, 1, size=IMAGE_SHAPE)
    im[im > 1] = 1
    im[im < -1] = -1
    return im

# this is the average squared difference between the layer outputs
def calculate_content_loss(content_layers, target_layers):
    differences = []
    for i in range(len(content_layers)):
        if i == 4:
            content, target = content_layers[i], target_layers[i]
            differences.append(torch.mean((content - target)**2))
    return sum(differences)

def calculate_style_loss(style_layers, target_layers):
    # compute the Gram matrix - the auto-correlation of each filter activation
    layer_expectations = []
    for l in range(len(style_layers)):
        # skip conv4_2
        if l == 4:
            continue
        style_layer = style_layers[l]
        target_layer = target_layers[l]
        _, N, y, x = style_layer.data.size()
        M = y * x
        style_layer = style_layer.view(N, M)
        target_layer = target_layer.view(N, M)
        G_s = torch.mm(style_layer, style_layer.t())
        G_t = torch.mm(target_layer, target_layer.t())
        difference = torch.mean(((G_s - G_t) ** 2)/(M*N*2))
        normalized_difference = 0.2*(difference)
        layer_expectations.append(normalized_difference)
    return sum(layer_expectations)

def construct_image(content, style):
    if CLONE_CONTENT:
        target = Variable(content.clone().data, requires_grad=True)
    elif CLONE_STYLE:
        target = Variable(style.clone().data, requires_grad=True)
    else:
        target = Variable(torch.randn([1, 3, IM_SIZE, IM_SIZE]).type(TENSOR_TYPE), requires_grad=True)
    # NOTE: Experiment with learning rate later
    ## taken from pytorch docs: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    optimizer = LBFGS([target])
    vgg_activations = VGGActivations()
    if USE_CUDA:
        vgg_activations = vgg_activations.cuda()
    vgg = vgg_activations
    content_layers = vgg.forward(content)
    style_layers = vgg.forward(style)
    for i in range(N_ITER):
        target.data.clamp(0, 1)
        target_layers = vgg.forward(target)
        def closure():
            # zero gradient buffer to prevent buildup
            optimizer.zero_grad()
            style_loss = calculate_style_loss(style_layers, target_layers)
            content_loss = calculate_content_loss(content_layers, target_layers)
            if i % 10 == 0:
                print('Step:', i, '/', N_ITER)
                print('Style loss:', style_loss.data[0])
                print('Content loss:', content_loss.data[0])
            loss = content_loss * CONTENT_WEIGHT + style_loss * STYLE_WEIGHT
            content_loss.backward(retain_graph=True)
            style_loss.backward(retain_graph=True)
            if (i+1) % 100 == 0:
                if USE_CUDA:
                    cloned_param = target.clone().cpu()
                else:
                    cloned_param = target.clone()
                print(cloned_param.data.size())
                im = cloned_param.squeeze(0).data
                denorm = trans.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
                utils.save_image(denorm(im).clamp(0,1), OUT_PATH + DATASET + '-' + str(STYLE_WEIGHT) + '_' + str(CONTENT_WEIGHT) + 'output_' + str(i) + '.'+ F_EXT)
            return loss
        optimizer.step(closure)

if __name__ == "__main__":
    # if len(sys.argv) > 1 and sys.argv[1] == '--gpu':
    #     USE_CUDA = True
    USE_CUDA = torch.cuda.is_available()
    print('using gpu:', USE_CUDA)
    if USE_CUDA:
        TENSOR_TYPE = torch.cuda.FloatTensor
    content, style = load_images()
    final_image = construct_image(content, style)
