import sys
import torch
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

IM_PATH = 'input/'
OUT_PATH = 'output/'
F_EXT = 'JPG'
CONTENT_IMAGE = IM_PATH + 'content.jpg'
STYLE_IMAGE = IM_PATH + 'style.jpg'
IM_SIZE = 128
IMAGE_SHAPE = (IM_SIZE, IM_SIZE, 3)
USE_CUDA = False
STYLE_WEIGHT = 1000.
CONTENT_WEIGHT = 10.
N_ITER = 500
STYLE_LAYER_WEIGHTS = [0.2 for _ in range(5)]
TENSOR_TYPE = torch.FloatTensor

class VGGActivations(nn.Module):
    def __init__(self):
        super(VGGActivations, self).__init__()
        self.vgg = models.vgg19(pretrained=True)

    def forward(self, x):
        conv_results = []
        for i, layer in enumerate(self.vgg.features):
            x = layer(x)
            if type(layer) == torch.nn.modules.conv.Conv2d:
                conv_results.append(x)
            if len(conv_results) == 5:
                break
        return conv_results


def toTorch(im):
    im = sktrans.resize(im, IMAGE_SHAPE, mode='constant')
    im = Variable(trans.ToTensor()(im))
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
        if i == 3:
            content, target = content_layers[i], target_layers[i]
            differences.append(torch.mean((content - target)**2))
    return differences[0]

def calculate_style_loss(style_layers, target_layers):
    # compute the Gram matrix - the auto-correlation of each filter activation
    layer_expectations = []
    for l in range(len(style_layers)):
        style_layer = style_layers[l]
        target_layer = target_layers[l]
        _, N, y, x = style_layer.data.size()
        M = y * x
        style_layer = style_layer.view(N, M)
        target_layer = target_layer.view(N, M)
        G_s = torch.mm(style_layer, style_layer.t())
        G_t = torch.mm(target_layer, target_layer.t())
        difference = torch.mean(((G_s - G_t) ** 2)/(M*N*2))
        normalized_difference = STYLE_LAYER_WEIGHTS[l]*(difference)
        layer_expectations.append(normalized_difference)
    return sum(layer_expectations)

def construct_image(content, style):
    target = Variable(torch.randn([1, 3, IM_SIZE, IM_SIZE]).type(TENSOR_TYPE), requires_grad=True)
    # NOTE: Experiment with learning rate later
    optimizer = LBFGS([target])
    vgg_activations = VGGActivations()
    if USE_CUDA:
        vgg_activations = vgg_activations.cuda()
    vgg = vgg_activations
    content_layers = vgg.forward(content)
    style_layers = vgg.forward(style)
    # pdb.set_trace()
    for i in range(N_ITER):
        # zero gradient buffer to prevent buildup
        target_layers = vgg.forward(target)
        def closure():
            optimizer.zero_grad()
            style_loss = calculate_style_loss(style_layers, target_layers)
            content_loss = calculate_content_loss(content_layers, target_layers)
            if i % 10 == 0:
                print('Step:', i, '/', N_ITER)
                print('Style loss:', style_loss.data[0])
                print('Content loss:', content_loss.data[0])
            loss = content_loss * CONTENT_WEIGHT + style_loss * STYLE_WEIGHT
            loss.backward(retain_graph=True)
            if i % 100 == 0:
                if USE_CUDA:
                    cloned_param = target.clone().cpu()
                else:
                    cloned_param = target.clone()
                print(cloned_param.data.size())
                im = cloned_param.squeeze(0).data.numpy()
                # reshape the image to be (N, N, 3) from (3, N, N)
                im = np.moveaxis(im, 0, -1)
                # print('range of values: ', np.min(im), np.max(im))
                # print('Saved image with shape:', im.shape)
                skio.imsave(OUT_PATH + 'output_' + str(i) + '.'+ F_EXT, im)
            return loss


        optimizer.step(closure)
    return target_param.squeeze(0).data.numpy()

if __name__ == "__main__":
    # if len(sys.argv) > 1 and sys.argv[1] == '--gpu':
    #     USE_CUDA = True
    USE_CUDA = torch.cuda.is_available()
    print('using gpu:', USE_CUDA)
    if USE_CUDA:
        TENSOR_TYPE = torch.cuda.FloatTensor
    content, style = load_images()
    final_image = construct_image(content, style)
    plt.imshow(final_image)
    plt.show()
