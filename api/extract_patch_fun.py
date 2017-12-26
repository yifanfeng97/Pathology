import slide_fun
import config_fun
import random
import numpy as np
import scipy.ndimage as ndimage
from PIL import Image
from skimage.morphology import dilation, star
from skimage.filters import threshold_otsu
from itertools import product
import os

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
        self._max_mask_size = np.ceil(self._img.level_dimensions[0]/self._cfg.max_frac)
        self._max_mask_level = None

    def _get_level(self, size):
        level = self._img.level_count -1
        while self._img.level_dimensions[level][0] < size[0] and \
            self._img.level_dimensions[level][1] < size[0]:
            level -= 1
        return level

    def _threshold_downsample_level(self, img):
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

    def _merge_mask_files(self):
        selected_mask = np.zeros((self._max_mask_size[1], self._max_mask_size[0]), np.uint8)
        tumor_mask = np.zeros((self._max_mask_size[1], self._max_mask_size[0]), np.uint8)
        for mask_file in self._mask_files:
            anno = slide_fun.get_mask_info(os.path.basename(mask_file.split('.')[0]))
            level = int(anno[1])
            origin = (int(anno[2]), int(anno[3]))
            size = (int(anno[4]), int(anno[5]))

            # read annotation file
            with open(mask_file, 'rb') as f:
                mask_data = f.read()
                mask_data = np.frombuffer(mask_data, np.uint8)
                mask_data = mask_data.reshape([size[1], size[0]])

            new_origin = origin
            new_size = size
            factor = 1
            new_mask = Image.fromarray(mask_data)

            anno_level_size = self._img.level_dimensions[level]
            if anno_level_size[0] != self._max_mask_size[0] and \
                            anno_level_size[1] != self._max_mask_size[1]:
                factor = self._max_mask_size[0] / float(anno_level_size[0])

                new_size = [int(np.ceil(size[0] * factor)),
                            int(np.ceil(size[1] * factor))]
                new_origin = [int(np.ceil(origin[0] * factor)),
                              int(np.ceil(origin[1] * factor))]
                new_mask = new_mask.resize(new_size)

            selected_mask[new_origin[1]: new_size[1] + new_origin[1],
                            new_origin[0]: new_size[0] + new_origin[0]] = SELECTED
            new_mask = np.asarray(new_mask)
            tumor_mask[new_origin[1]: new_size[1] + new_origin[1],
                        new_origin[0]: new_size[0] + new_origin[0]] = new_mask
        return selected_mask, tumor_mask


    def _generate_mask(self):

        # init mask without background
        self._min_mask = None
        self._min_mask_level = self._get_level((2048, 2048))
        self._min_mask_size = self._img.level_dimensions[self._min_mask_level]

        self._max_mask_level = self._get_level(self._max_mask_size)
        self._max_mask = np.zeros((self._max_mask_size[1], self._max_mask_size[0]), np.uint8)
        th_img = self._img.read_region((0, 0), self._max_mask_level,
                                      self._img.level_dimensions[self._max_mask_level])
        th_img = th_img.resize(self._max_mask_size)
        th_mask = self._threshold_downsample_level(th_img)

        assert self._max_mask_size == (th_mask.size[1], th_mask.size[0])

        if self._mask_files is not None:
            selected_mask, tumor_mask = self._merge_mask_files()
            normal_and = np.logical_and(th_mask, selected_mask)

            self._max_mask[selected_mask !=0] = SELECTED
            self._max_mask[normal_and != 0] = NORMAL
            self._max_mask[tumor_mask != 0] = TUMOR
        else:
            self._max_mask[th_mask != 0] = NORMAL

        self._max_mask = Image.fromarray(self._max_mask)
        self._min_mask = self._max_mask.resize(self._min_mask_size)
        self._min_mask = np.asarray(self._min_mask)

        if self._cfg.vis_ov_mask:
            pass



    def _save_random_mask_and_patch(self):
        pass

    def _get_train_patch(self):
        patches = {'pos': [], 'neg': []}

        pass

        return patches

    def _save_patch(self):
        pass



def extract(data, type, auto_save_patch = True):
    img = single_img_process(data, type, auto_save_patch)
    img._generate_mask()
    img._get_train_patch()