from __future__ import print_function, division
import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import dilation, star, erosion, disk, opening
from scipy import ndimage
import glob
from PIL import Image
import os
from tqdm import tqdm
import time

data_folder = '../data'
result_folder = '../data/result'
SELECTED = 1
BACKGROUND = 0

def _threshold_downsample_level( img):
    """Generates thresholded overview image.

    Args:
        wsi: An openslide image instance.

    Returns:
        A 2D numpy array of binary image
    """

    # calculate the overview level size and retrieve the image
    img_hsv = img.convert('HSV')
    img_hsv_np = np.array(img_hsv)

    # dilate image and then threshold the image
    schannel = img_hsv_np[:, :, 1]
    mask = np.zeros(schannel.shape).astype(np.uint8)
    schannel = dilation(schannel, star(3))
    schannel = ndimage.gaussian_filter(schannel, sigma=(5, 5), order=0)
    threshold_global = threshold_otsu(schannel)

    # schannel[schannel > threshold_global] = 255
    # schannel[schannel <= threshold_global] = 0
    mask[schannel > threshold_global] = SELECTED
    mask[schannel <= threshold_global] = BACKGROUND

    # import scipy.misc   # check the result
    # scipy.misc.imsave('outfile.jpg', schannel)

    return mask

def seg(img):
    img_np = np.asarray(img)
    img_np_g = img_np[:, :, 1]
    shape = img_np_g.shape
    mask = np.ones(shape).astype(np.uint8)
    searched = np.zeros((shape)).astype(np.uint8)
    coor = []
    init_val = 0
    def inRange(val):
        return val >= init_val-10 and val <= init_val+10
    def addSeed_initVal():
        sums = 0
        for idx in range(shape[0]):
            coor.append({'x': idx, 'y': 0})
            searched[idx, 0] = 1
            sums += img_np_g[idx, 0]
        return sums/shape[0]

    def isPixel(x, y):
        return (x>=0 and x<shape[0]) and (y>=0 and y<shape[1])
    def deal(x, y):
        if isPixel(x, y) and not searched[x, y] and inRange(img_np_g[x, y]):
            coor.append({'x': x, 'y': y})
            searched[x, y] = 1
            mask[x, y] = 0
    init_val = addSeed_initVal()
    print('init val: %d'%init_val)

    while coor != []:
        x = coor[0]['x']
        y = coor[0]['y']
        if x == 0: deal(x, y)
        del coor[0]
        deal(x+1, y)
        deal(x, y+1)
        deal(x-1, y)
        deal(x, y-1)

    # mask = opening(mask, star(5))
    # mask = erosion(mask, star(3))
    return mask

if __name__ == '__main__':
    # flag = ' dil_gau5_otsu'
    flag = ' seg_open'
    file_names = glob.glob(os.path.join(data_folder, '*.png'))
    # file_names = ['../data/2017-03225-1_2017-07-27 13_57_55_thumb.png']
    if not os.path.exists(result_folder):
        os.mkdir(result_folder)

    for idx, file_name in enumerate(tqdm(file_names)):
        print('file name:'+file_name)
        img = Image.open(file_name)
        start_t = time.time()
        mask = _threshold_downsample_level(img)
        # mask = seg(img)
        print('use time: %.2fs'% (time.time()-start_t))
        mask_img = Image.fromarray(mask.astype(np.uint8)*255)
        img.save(os.path.join(result_folder,
            os.path.basename(file_name)[:-4] + '.jpg'))
        mask_img.save(os.path.join(result_folder,
            os.path.basename(file_name)[:-4] + flag + '.jpg'))
        img.close()


