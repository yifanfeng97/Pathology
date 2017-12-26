import slide_fun
import config_fun
import random
import numpy as np
import scipy.ndimage as ndimage
from PIL import Image
from skimage.morphology import dilation, star
from skimage.filters import threshold_otsu
from itertools import product

BACKGROUND = 0
SELECTED = 1
NORMAL = 2
TUMOR = 3


class single_img_process():
    def __init__(self, data, type, auto_save_patch = True):
        self._cfg = config_fun.config()
        self._file_name = data['data'][0]
        self._mask_files = data['data'][1]
        self._auto_save_patch = auto_save_patch
        self._type = type

        self._img = slide_fun.AllSlide(self._file_name)
        self._max_mask = None
        self._max_mask_size = self._img.level_dimensions[0]

    def _get_level(self, size):
        level = self._img.level_count -1
        while self._img.level_dimensions[level][0] < size and \
            self._img.level_dimensions[level][1] < size:
            level -= 1
        return level

    def _threshold_downsample_lvl(self, img):
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
        schannel = dilation(schannel, star(3))
        schannel = ndimage.gaussian_filter(schannel, sigma=(5, 5), order=0)
        threshold_global = threshold_otsu(schannel)

        schannel[schannel > threshold_global] = SELECTED
        schannel[schannel <= threshold_global] = BACKGROUND

        # import scipy.misc   # check the result
        # scipy.misc.imsave('outfile.jpg', schannel)

        return schannel

    def _generate_mask(self):
        self._min_mask = None
        self._min_mask_size = self._get_level(2048)

        self._max_mask = self._threshold_downsample_lvl()


    def _save_random_mask_and_patch(self):
        pass

    def _get_train_patch(self):
        pass

    def _save_patch(self):
        pass



def extract(data, type, auto_save_patch = True):
    img = single_img_process(data, type, auto_save_patch)
    img._generate_mask()
    img._get_train_patch()