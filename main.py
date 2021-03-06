from skimage import data, segmentation, color, io, exposure
from minisom import MiniSom
import numpy as np
import argparse

# get color histogram of each superpixel, no normalized
def get_color_histogram(image, superpixels, index):
    indices = np.where(superpixels.ravel() == index)[0]
    _r_hist = np.bincount(image[:, :, 0].ravel()[indices], minlength=256)
    _g_hist = np.bincount(image[:, :, 1].ravel()[indices], minlength=256)
    _b_hist = np.bincount(image[:, :, 2].ravel()[indices], minlength=256)

    return _r_hist, _g_hist, _b_hist

# get grayscale histogram of each superpixel, not normalized
def get_grayscale_histogram(image, superpixels, index):
    indices = np.where(superpixels.ravel() == index)[0]
    _gray_hist = np.bincount(image[:, :].ravel()[indices], minlength=256)

    return _gray_hist

parser = argparse.ArgumentParser()
parser.add_argument('input_image', type=str, help='input image path')
parser.add_argument('num_superpixel', type=int, help='number of segments')
parser.add_argument('compactness', type=int, help='compactness param of SLIC')
args = parser.parse_args()


img = data.coffee()
#img = io.imread(args.input_image)
labels = segmentation.slic(img, n_segments=args.num_superpixel, compactness=args.compactness)
out1 = color.label2rgb(labels, img, kind='avg')
#print(labels.shape)
#print(np.unique(labels))

io.imshow(out1)
io.show()

out1_gray = color.rgb2gray(out1)
io.imshow(out1_gray)
io.show()

hist = np.zeros((np.max(labels), 256), dtype=float)
#print(hist.shape)
for i in range(np.max(labels)):
    gray_hist = get_grayscale_histogram(img, labels, i)
    gray_hist = np.float32(gray_hist / gray_hist.sum())
    hist[i, :] = gray_hist

print(hist.shape)
'''
for i in range(np.max(labels)):
    r_hist, g_hist, b_hist = get_color_histogram(img, labels, i)
    hist[i][0] = r_hist
    hist[i][1] = g_hist
    hist[i][2] = b_hist
    #hist[i][0], hist[i][1], hist[i][2] = get_color_histogram(img, labels, i)
'''
print('training...')
som = MiniSom(5, 5, 256, sigma=0.5, learning_rate=0.2, neighborhood_function='gaussian')
som.random_weights_init(hist)
starting_weights = som.get_weights().copy()  # saving the starting weights
som.train_random(hist, 1000, verbose=True)

#print('quantization...')
#qnt = som.quantization(hist)  # quantize each pixels of the image
#print('building new image...')
print('done.')
print(starting_weights.shape)
io.imshow(starting_weights)
io.show()
io.imshow(som.get_weights())
io.show()
