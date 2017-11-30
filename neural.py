import torch
import skimage.io as skio
import skimage.transform as sktr
import skimage.filters as skfltr

IM_PATH = 'input/'
F_EXT = 'JPG'
CONTENT_IMAGE = IM_PATH + 'content.jpg'
STYLE_IMAGE = IM_PATH + 'style.jpg'
def load_images():
    content = skio.imread(CONTENT_IMAGE)
    style = skio.imread(STYLE_IMAGE)
    # consider resizing in a more intelligent way
    optimal_sigma = (1 - (content.shape[0] / style.shape[0])) / 2
    print('Filter sigma:', optimal_sigma)
    style = sktr.resize(skfltr.gaussian(style, s=optimal_sigma), content.shape)
    assert style.shape == content.shape, "Image shapes are not equal"
    return content, style

def initialize_target_image(shape):
    im = np.zeros(shape)
    return skimage.util.random_noise(im)

content, style = load_images()
plt.imshow(content)
plt.show()
plt.imshow(style)
plt.show()
plt.imshow(initialize_target_image(style.shape))
plt.show()
