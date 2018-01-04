import slide_fun
import config_fun
import random
import numpy as np
import scipy.ndimage as ndimage
from PIL import Image
from skimage.morphology import dilation, star, opening, erosion
from skimage.filters import threshold_otsu
from itertools import product
import os

BACKGROUND = 0
SELECTED = 1
NORMAL = 2
TUMOR = 3

SELECTED_COLOR = [0, 0, 255] # Blue
NORMAL_COLOR = [0, 255, 0] # Green
TUMOR_COLOR = [255, 0, 0] # Red


class single_img_process():
    def __init__(self, file_name, mask_files, file_type, patch_type, auto_save_patch = True):
        self._cfg = config_fun.config()
        self._file_name = file_name
        self._mask_files = mask_files
        self._auto_save_patch = auto_save_patch
        self._file_type = file_type
        self._patch_type = patch_type

        self._img = slide_fun.AllSlide(self._file_name)
        self._max_mask = None
        self._max_mask_size = np.ceil(np.array(self._img.level_dimensions[0])/self._cfg.max_frac).astype(np.int)
        self._max_mask_level = None

        self._min_patch_size = int(self._cfg.patch_size/self._cfg.min_frac)

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
        mask = np.zeros(schannel.shape)
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

    def _seg_dfs(self, img):
        img_np = np.asarray(img)
        img_np_g = img_np[:, :, 1]
        shape = img_np_g.shape
        mask = np.ones(shape).astype(np.uint8)
        searched = np.zeros((shape)).astype(np.uint8)
        coor = []
        init_val = 0

        def inRange(val):
            return val >= init_val - 10 and val <= init_val + 10

        def addSeed_initVal():
            sums = 0
            for idx in range(shape[0]):
                coor.append({'x': idx, 'y': 0})
                searched[idx, 0] = 1
                sums += img_np_g[idx, 0]
            return sums / shape[0]

        def isPixel(x, y):
            return (x >= 0 and x < shape[0]) and (y >= 0 and y < shape[1])

        def deal(x, y):
            if isPixel(x, y) and not searched[x, y] and inRange(img_np_g[x, y]):
                coor.append({'x': x, 'y': y})
                searched[x, y] = 1
                mask[x, y] = 0

        init_val = addSeed_initVal()
        # print('init val: %d' % init_val)

        while coor != []:
            x = coor[0]['x']
            y = coor[0]['y']
            if x == 0: deal(x, y)
            del coor[0]
            deal(x + 1, y)
            deal(x, y + 1)
            deal(x - 1, y)
            deal(x, y - 1)

        mask = opening(mask, star(5))
        mask = erosion(mask, star(3))
        return mask

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

    def _generate_img_bg_mask(self):
        self._max_mask_level = self._get_level(self._max_mask_size)
        self._max_mask = np.zeros((self._max_mask_size[1], self._max_mask_size[0]), np.uint8)
        th_img = self._img.read_region((0, 0), self._max_mask_level,
                                       self._img.level_dimensions[self._max_mask_level])
        th_img = th_img.resize(self._max_mask_size)
        # th_mask = self._threshold_downsample_level(th_img)
        th_mask = self._seg_dfs(th_img)

        return th_img, th_mask

    def _generate_mask(self):

        # init mask without background
        self._min_mask = None
        self._min_mask_size = np.ceil(np.array(self._img.level_dimensions[0])/self._cfg.min_frac).astype(np.int)
        self._min_mask_size = (np.ceil(self._img.level_dimensions[0][0] / self._cfg.min_frac),
                               np.ceil(self._img.level_dimensions[0][1] / self._cfg.min_frac))
        self._min_mask_level = self._get_level(self._min_mask_size)

        th_img, th_mask = self._generate_img_bg_mask()
        th_img.close()

        # Image.fromarray(th_mask * 255).show()
        # th_img.save(os.path.join(self._cfg.vis_ov_mask_folder, os.path.basename(
        #         self._file_name)[:-4] + '.png'))

        assert (self._max_mask_size[1], self._max_mask_size[0]) == th_mask.shape

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
        self._max_mask = np.asarray(self._max_mask)

        if self._cfg.vis_ov_mask:
            raw_img = self._img.read_region((0, 0), self._min_mask_level,
                                            self._img.level_dimensions[self._min_mask_level])
            raw_img = raw_img.resize(self._min_mask_size)

            mask_img = np.zeros((raw_img.size[1], raw_img.size[0], 3), np.uint8)
            mask_img = Image.fromarray(mask_img)
            mask = self._min_mask.copy()
            assert (raw_img.size[1], raw_img.size[0]) == mask.shape

            raw_mask_img = self._fusion_mask_img(raw_img, mask)
            mask_img = self._fusion_mask_img(mask_img, mask)

            raw_img.save(os.path.join(self._cfg.vis_ov_mask_folder, os.path.basename(
                self._file_name)[:-4] + '_raw' + self._cfg.img_ext))
            raw_mask_img.save(os.path.join(self._cfg.vis_ov_mask_folder, os.path.basename(
                self._file_name)[:-4] + '_raw_mask' + self._cfg.img_ext))
            mask_img.save(os.path.join(self._cfg.vis_ov_mask_folder, os.path.basename(
                self._file_name)[:-4] + '_mask' + self._cfg.img_ext))

            mask_img.close()
            raw_mask_img.close()
            raw_img.close()


    def _fusion_mask_img(self, img, mask):
        # mask type array
        # img type array
        img_np = np.asarray(img)
        assert (img.size[1], img.size[0]) == mask.shape
        img_mask = img_np.copy()
        if (mask == TUMOR).any():
            img_mask[mask == TUMOR] = self._cfg.alpha * img_np[mask == TUMOR] + \
                                      (1 - self._cfg.alpha) * np.array(TUMOR_COLOR)
        if (mask == NORMAL).any():
            img_mask[mask == NORMAL] = self._cfg.alpha * img_np[mask == NORMAL] + \
                                       (1 - self._cfg.alpha) * np.array(NORMAL_COLOR)
        if (mask == SELECTED).any():
            img_mask[mask == SELECTED] = self._cfg.alpha * img_np[mask == SELECTED] + \
                                         (1 - self._cfg.alpha) * np.array(SELECTED_COLOR)
        return Image.fromarray(img_mask)

    def _is_bg(self, origin):
        img = self._img.read_region(origin, 0, (self._cfg.patch_size, self._cfg.patch_size))
        # bad case is background    continue
        if np.array(img)[:, :, 1].mean() > 200: # is bg
            img.close()
            return True
        else:
            img.close()
            return False

    def _save_random_patch(self, origin, min_patch):
        if random.random()>self._cfg.vis_patch_prob:
            return
        img = self._img.read_region(origin, 0, (self._cfg.patch_size, self._cfg.patch_size))

        # max_patch_origin = (np.array(origin)/self._cfg.max_frac).astype(np.int)
        # max_patch_size = int(self._cfg.patch_size/self._cfg.max_frac)
        # mask = self._max_mask[max_patch_origin[0]: max_patch_origin[0] + max_patch_size,
        #                         max_patch_origin[1]: max_patch_origin[1] + max_patch_size]
        mask = min_patch
        mask = Image.fromarray(mask)
        mask = mask.resize((self._cfg.patch_size, self._cfg.patch_size))

        mask = np.asarray(mask)
        img_mask = self._fusion_mask_img(img, mask)

        if self._patch_type == 'pos':
            img_mask.save(os.path.join(self._cfg.vis_pos_patch_folder,
                                  os.path.basename(self._file_name).split('.')[0]
                                  + '_%d_%d' % (origin[0], origin[1]) + self._cfg.img_ext))
        else:
            img_mask.save(os.path.join(self._cfg.vis_neg_patch_folder,
                                  os.path.basename(self._file_name).split('.')[0]
                                  + '_%d_%d' % (origin[0], origin[1]) + self._cfg.img_ext))
        img.close()
        img_mask.close()

    def _save_patches(self, patches):

        cnt = 0
        if patches['pos'] == []:
            patches = patches['neg']
        else:
            patches = patches['pos']
        random.shuffle(patches)
        for patch in patches:
            if cnt >= self._cfg.patch_num_in_train:
                break
            img = self._img.read_region(patch, 0, (self._cfg.patch_size, self._cfg.patch_size))
            folder_pre = None
            if self._file_type == 'train':
                folder_pre = os.path.join(self._cfg.patch_save_folder, 'train')
            else:
                folder_pre = os.path.join(self._cfg.patch_save_folder, 'val')
            self._cfg.check_dir(folder_pre)
            if self._patch_type == 'pos':
                folder_pre = os.path.join(folder_pre, 'pos')
            else:
                folder_pre = os.path.join(folder_pre, 'neg')
            self._cfg.check_dir(folder_pre)

            img.save(os.path.join(folder_pre, os.path.basename(self._file_name)[:-4]
                                  + '_%d_%d' % patch + self._cfg.img_ext))
            img.close()
            cnt +=1

    def _get_train_patch(self):
        do_bg_filter = False
        patches = {'pos': [], 'neg': []}
        assert self._min_mask_size[1], self._min_mask_size[0] == self._min_mask.shape
        num_row, num_col = self._min_mask_size
        num_row = num_row - self._min_patch_size
        num_col = num_col - self._min_patch_size

        # step = 1
        row_col = list(product(range(num_row), range(num_col)))
        random.shuffle(row_col)
        cnt = 0
        for row, col in row_col:
            if cnt >= self._cfg.patch_num_in_file:
                break
            min_patch = self._min_mask[row: row + self._min_patch_size,
                       col: col + self._min_patch_size]
            origin = (int(col * self._cfg.min_frac), int(row * self._cfg.min_frac))

            H, W = min_patch.shape
            H_min = int(np.ceil(H / 4))
            H_max = int(np.ceil(H / 4 * 3))
            W_min = int(np.ceil(W / 4))
            W_max = int(np.ceil(W / 4 * 3))
            # half of the center
            th_num = int(np.ceil((H/2 * W/2) /2))
            if self._patch_type == 'pos':
                if np.count_nonzero(min_patch[H_min:H_max, W_min:W_max] == TUMOR) > th_num:
                    if do_bg_filter:
                        if self._is_bg(origin):
                            continue
                    patches['pos'].append(origin)
                    self._save_random_patch(origin, min_patch)
                    cnt+=1

            if self._patch_type == 'neg':
                if np.count_nonzero(min_patch[H_min:H_max, W_min:W_max] == NORMAL) > th_num:
                    if do_bg_filter:
                        if self._is_bg(origin):
                            continue
                    patches['neg'].append(origin)
                    self._save_random_patch(origin, min_patch)
                    cnt+=1
        if self._auto_save_patch:
            self._save_patches(patches)
        return patches

    def _save_patch(self):
        pass



def extract(data, file_type, patch_type, auto_save_patch = True):
    img = single_img_process(data['data'][0], data['data'][1], file_type, patch_type, auto_save_patch)
    img._generate_mask()
    return img._get_train_patch()

if __name__ == '__main__':
    pass