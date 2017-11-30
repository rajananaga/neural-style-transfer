import torch
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
def load_images():
    content = skio.imread(CONTENT_IMAGE)
    style = skio.imread(STYLE_IMAGE)
    # consider resizing in a more intelligent way
    optimal_sigma = 0.2
    print('Filter sigma:', optimal_sigma)
    style = sktr.resize(skfltr.gaussian(style, sigma=optimal_sigma, multichannel=True), content.shape, mode='constant')
    assert style.shape == content.shape, "Image shapes are not equal"
    return content, style

def initialize_target_image(shape):
    im = np.zeros(shape)
    return skutil.random_noise(im)

content, style = load_images()
plt.imshow(content)
plt.show()
plt.imshow(style)
plt.show()
plt.imshow(skclr.rgb2gray(initialize_target_image(style.shape)), cmap='gray')
plt.show()
