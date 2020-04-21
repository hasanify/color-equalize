from IPython.display import display, Math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img = Image.open('sample.jpg')
plt.figure(num="Original Image")
plt.imshow(img)

img = np.asarray(img)
flat = img.flatten()

display(Math(r'P_x(j) = \sum_{i=0}^{j} P_x(i)'))

def get_histogram(image, bins):
    histogram = np.zeros(bins)
    
    for pixel in image:
        histogram[pixel] += 1

    return histogram

hist = get_histogram(flat, 256)

def sum(a):
    a = iter(a)
    b = [next(a)]
    for i in a:
        b.append(b[-1] + i)
    return np.array(b)

cs = sum(hist)

display(Math(r's_k = \sum_{j=0}^{k} {\frac{n_j}{N}}'))

nj = (cs - cs.min()) * 255
N = cs.max() - cs.min()

cs = nj / N
cs = cs.astype('uint8')

img_new = cs[flat]
img_new = np.reshape(img_new, img.shape)
img_new
plt.figure(num="Equalized Color Image")
plt.imshow(img_new)

plt.show(block=True)