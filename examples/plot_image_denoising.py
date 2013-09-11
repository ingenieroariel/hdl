#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from time import time

import pylab as pl
import numpy as np

from scipy.misc import lena as get_lena

from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d

###############################################################################
# Load Lena image and extract patches

lena = get_lena() / 256.0
height, width = lena.shape

# Distort the the image
print('Distorting image...')
lena_copy = lena.copy()

from PIL import Image, ImageDraw, ImageFont
import numpy
im = Image.fromarray((lena_copy*256).astype(numpy.uint8))
#TEXT = 'En realidad, Remedios, la bella, no era un ser de este mundo. Hasta muy avanzada la pubertad, Santa Sofía de la Piedad tuvo que bañarla y ponerle la ropa, y aún cuando pudo valerse por sí misma había que vigilarla para que no pintara animalitos en las paredes con una varita embadurnada de su propia caca. Llegó a los veinte años sin aprender a leer y escribir, sin servirse de los cubiertos en la mesa, paseándose desnuda por la casa, porque su naturaleza se resistía a cualquier clase de convencionalismos'
TEXT = 'Muchos años después, frente al pelotón de fusilamiento, el coronel Aureliano Buendía había de recordar aquella tarde remota en que su padre lo llevó a conocer el hielo'
font_path = '/Library/Fonts/Arial.ttf'
font = ImageFont.truetype(font_path, 42, encoding='unic')
text = TEXT.decode('utf-8')
draw = ImageDraw.Draw(im)

import textwrap
h, w = 25, 34
lines = textwrap.wrap(text, width = w)
y_text = h
x_text = 0
for line in lines:
    width2, height2 = font.getsize(line)
    draw.text((x_text, y_text), line, font = font, fill = 'white')
    y_text += height2 * 2

im.save('distorted.png')

from scipy import misc
#distorted = misc.imread('distorted.png') / 256.0
distorted = numpy.asarray(im) / 256.0

# Extract all reference patches from the left half of the image
print('Extracting reference patches...')
t0 = time()
patch_size = (8, 8)
data = extract_patches_2d(distorted[:, :height / 2], patch_size)
data = data.reshape(data.shape[0], -1)
data -= np.mean(data, axis=0)
data /= np.std(data, axis=0)
print('done in %.2fs.' % (time() - t0))

###############################################################################
# Learn the dictionary from reference patches

print('Learning the dictionary...')
t0 = time()
dico = MiniBatchDictionaryLearning(n_components=100, alpha=1, n_iter=500)
V = dico.fit(data).components_
dt = time() - t0
print('done in %.2fs.' % dt)

pl.figure(figsize=(4.2, 4))
for i, comp in enumerate(V[:100]):
    pl.subplot(10, 10, i + 1)
    pl.imshow(comp.reshape(patch_size), cmap=pl.cm.gray_r,
              interpolation='nearest')
    pl.xticks(())
    pl.yticks(())
pl.suptitle('Dictionary learned from Lena patches\n' +
            'Train time %.1fs on %d patches' % (dt, len(data)),
            fontsize=16)
pl.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)


###############################################################################
# Display the distorted image

def show_with_diff(image, reference, title):
    """Helper function to display denoising"""
    pl.figure(figsize=(5, 3.3))
    pl.subplot(1, 2, 1)
    pl.title('Image')
    pl.imshow(image, vmin=0, vmax=1, cmap=pl.cm.gray, interpolation='nearest')
    pl.xticks(())
    pl.yticks(())
    pl.subplot(1, 2, 2)
    difference = image - reference

    pl.title('Difference (norm: %.2f)' % np.sqrt(np.sum(difference ** 2)))
    pl.imshow(difference, vmin=-0.5, vmax=0.5, cmap=pl.cm.PuOr,
              interpolation='nearest')
    pl.xticks(())
    pl.yticks(())
    pl.suptitle(title, size=16)
    pl.subplots_adjust(0.02, 0.02, 0.98, 0.79, 0.02, 0.2)

show_with_diff(distorted, lena, 'Distorted image')

###############################################################################
# Extract noisy patches and reconstruct them using the dictionary

print('Extracting noisy patches... ')
t0 = time()
data = extract_patches_2d(distorted[:, height / 2:], patch_size)
data = data.reshape(data.shape[0], -1)
intercept = np.mean(data, axis=0)
data -= intercept
print('done in %.2fs.' % (time() - t0))

transform_algorithms = [
    ('Orthogonal Matching Pursuit\n1 atom', 'omp',
     {'transform_n_nonzero_coefs': 1}),
    ('Orthogonal Matching Pursuit\n2 atoms', 'omp',
     {'transform_n_nonzero_coefs': 2}),
    ('Least-angle regression\n5 atoms', 'lars',
     {'transform_n_nonzero_coefs': 5}),
    ('Thresholding\n alpha=0.1', 'threshold', {'transform_alpha': .1})]

reconstructions = {}
for title, transform_algorithm, kwargs in transform_algorithms:
    print(title + '...')
    reconstructions[title] = lena.copy()
    t0 = time()
    dico.set_params(transform_algorithm=transform_algorithm, **kwargs)
    code = dico.transform(data)
    patches = np.dot(code, V)

    if transform_algorithm == 'threshold':
        patches -= patches.min()
        patches /= patches.max()

    patches += intercept
    patches = patches.reshape(len(data), *patch_size)
    if transform_algorithm == 'threshold':
        patches -= patches.min()
        patches /= patches.max()
    reconstructions[title][:, height / 2:] = reconstruct_from_patches_2d(
        patches, (width, height / 2))
    dt = time() - t0
    print('done in %.2fs.' % dt)
    show_with_diff(reconstructions[title], lena,
                   title + ' (time: %.1fs)' % dt)

pl.show()
