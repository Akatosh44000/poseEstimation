""" Radon Transform as described in Birkfellner, Wolfgang. Applied Medical Image Processing: A Basic Course. [p. 344] """
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt

def discrete_radon_transform(image, steps):
    R = np.zeros((steps, len(image)), dtype='float64')
    for s in range(steps):
        rotation = misc.imrotate(image, -s*180/steps).astype('float64')
        R[:,s] = sum(rotation)
    return R

# Read image as 64bit float gray scale
image = misc.imread('/home/akatosh/DATASETS/MULTITUDE/BREATHER/VRAC/TEST/output_359.82_35.52_342.63.png', flatten=True).astype('float64')
radon = discrete_radon_transform(image, 64)

# Plot the original and the radon transformed image
plt.subplot(1, 2, 1), plt.imshow(image, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.subplot(1, 2, 2), plt.imshow(radon, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.show()