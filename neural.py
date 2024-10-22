import sys
import torch
import torchvision.utils as utils
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as trans
from torch.optim import LBFGS
from torch.autograd import Variable
import skimage.io as skio
import skimage.util as skutil
import skimage.transform as sktrans
import numpy as np

DATASET = 'monet2'
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
N_ITER = 300
STYLE_LAYER_WEIGHTS = [0.2 for _ in range(5)]
TENSOR_TYPE = torch.FloatTensor
CLONE_STYLE = False
CLONE_CONTENT = True

layer_names = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4', 'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4', 'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4']
layers = {layer_names[i]:i for i in range(len(layer_names))}

STYLE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
CONTENT_LAYERS = ['conv5_2']

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
            if type(layer) == torch.nn.modules.conv.Conv2d:
                conv_results.append(x)
        return conv_results

def toTorch(im, content_shape):
    im = sktrans.resize(im, content_shape, mode='constant')
    # taken from pytorch docs: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    transform = trans.Compose([trans.ToTensor(),trans.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    im = Variable(transform(im))
    # VGG network throws error if the shape doesn't have a 1 in front (1 x 512 x 512)
    im = im.unsqueeze(0)
    return im.type(TENSOR_TYPE)

def load_images():
    content = skio.imread(CONTENT_IMAGE)/1.
    style = skio.imread(STYLE_IMAGE)/1.
    s = content.shape
    while s[0] > 1000 or s[1] > 1000:
        s = [int(s[0]/2), int(s[1]/2), s[2]]
    content = toTorch(content, s)
    style = toTorch(style, s)
    assert style.data.size() == content.data.size(), "Image shapes are not equal"
    return content, style

def initialize_target_image(shape):
    im = np.random.uniform(0, 1, size=shape)
    im[im > 1] = 1
    im[im < -1] = -1
    return im

# This is the average squared difference between the layer outputs
def calculate_content_loss(content_layers, target_layers):
    wanted_layers = [layers[l] for l in CONTENT_LAYERS]
    differences = []
    for i in wanted_layers:
        content, target = content_layers[i], target_layers[i]
        differences.append(torch.mean((content - target)**2))
    return sum(differences)

def calculate_style_loss(style_layers, target_layers):
    wanted_layers = [layers[l] for l in STYLE_LAYERS]
    layer_expectations = []
    for l in wanted_layers:
        style_layer = style_layers[l]
        target_layer = target_layers[l]
        _, N, y, x = style_layer.data.size()
        M = y * x
        style_layer = style_layer.view(N, M)
        target_layer = target_layer.view(N, M)
        # compute the Gram matrices - the auto-correlation of each filter activation
        G_s = torch.mm(style_layer, style_layer.t())
        G_t = torch.mm(target_layer, target_layer.t())
        # MSE of differences between the Gram matrices
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
        target = Variable(torch.randn([1, 3, content.data.size()[0], content.data.size()[1]]).type(TENSOR_TYPE), requires_grad=True)
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
                im = cloned_param.squeeze(0).data
                denorm = trans.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
                utils.save_image(denorm(im).clamp(0,1), OUT_PATH + DATASET + '-' + str(STYLE_WEIGHT) + '_' + str(CONTENT_WEIGHT) + 'output_' + str(i) + '.'+ F_EXT)
            return loss
        optimizer.step(closure)

if __name__ == "__main__":
    USE_CUDA = torch.cuda.is_available()
    print('using gpu:', USE_CUDA)
    print('using the following images:', 'style:', STYLE_IMAGE, 'content:', CONTENT_IMAGE)
    if USE_CUDA:
        TENSOR_TYPE = torch.cuda.FloatTensor
    content, style = load_images()
    final_image = construct_image(content, style)
