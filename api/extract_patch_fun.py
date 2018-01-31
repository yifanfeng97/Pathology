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
import sys

BACKGROUND = 0
SELECTED = 1
SAMPLED = 2
# NORMAL = 3
# TUMOR = 4

SELECTED_COLOR = [0, 0, 255] # Blue
NORMAL_COLOR = [0, 255, 0] # Green
TUMOR_COLOR = [255, 0, 0] # Red
# SAMPLED_COLOR = []


class single_img_process():
    def __init__(self, file_name, mask_files, file_type, patch_type, auto_save_patch = True):
        self._cfg = config_fun.config()
        self._file_name = file_name
        self._mask_files = mask_files
        self._auto_save_patch = auto_save_patch
        self._file_type = file_type
        self._patch_type = patch_type

        self._neg_start_idx = 3
        self._pos_start_idx = self._neg_start_idx + self._cfg.num_neg_classes

        self._img = slide_fun.AllSlide(self._file_name)
        self._max_mask = None
        self._max_mask_size = np.ceil(np.array(self._img.level_dimensions[0])/self._cfg.max_frac).astype(np.int)
        self._max_mask_level = None

        self._min_patch_size = int(self._cfg.patch_size/self._cfg.min_frac)

    def _get_level(self, size):
        level = self._img.level_count -1
        while level>=0 and self._img.level_dimensions[level][0] < size[0] and \
            self._img.level_dimensions[level][1] < size[1]:
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
        mask = np.ones(shape).astype(np.uint8) * SELECTED
        searched = np.zeros((shape)).astype(np.bool)
        coor = []
        init_val = 0

        def inRange(val):
            return val >= init_val - 10 and val <= init_val + 10

        def addSeed_initVal():
            val1 = img_np_g[:, 0].mean()
            val2 = img_np_g[0, :].mean()
            val3 = img_np_g[:, shape[1]-1].mean()
            val4 = img_np_g[shape[0]-1, 0].mean()
            val = np.max((val1, val2, val3, val4))
            for idx in range(shape[0]):
                # L
                coor.append({'x': idx, 'y': 0})
                searched[idx, 0] = True
                # R
                coor.append({'x': idx, 'y': shape[1]-1})
                searched[idx, shape[1]-1] = True
            for idx in range(shape[1]):
                # U
                coor.append({'x': 0, 'y': idx})
                searched[0, idx] = True
                # D
                coor.append({'x': shape[0]-1, 'y': idx})
                searched[shape[0]-1, idx] = True
            return val

        def isPixel(x, y):
            return (x >= 0 and x < shape[0]) and (y >= 0 and y < shape[1])

        def deal(x, y):
            if isPixel(x, y) and not searched[x, y] and inRange(img_np_g[x, y]):
                coor.append({'x': x, 'y': y})
                searched[x, y] = True
                mask[x, y] = BACKGROUND

        init_val = addSeed_initVal()
        # print('init val: %d' % init_val)

        while coor != []:
            x = coor[0]['x']
            y = coor[0]['y']
            if x == 0 or y == 0\
                or x == shape[0]-1 or y == shape[1]-1:
                deal(x, y)
            del coor[0]
            deal(x + 1, y)
            deal(x, y + 1)
            deal(x - 1, y)
            deal(x, y - 1)

        mask = opening(mask, star(5))
        # mask = erosion(mask, star(3))
        mask = dilation(mask, star(3))
        return mask

    def _merge_mask_files(self):
        selected_mask = np.zeros((self._max_mask_size[1], self._max_mask_size[0]), np.uint8)
        anno_mask = np.zeros((self._max_mask_size[1], self._max_mask_size[0]), np.uint8)
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
            # factor = 1
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
            if self._patch_type == 'pos':
                new_mask = new_mask[new_mask != 0] - 1 + self._pos_start_idx
            elif self._patch_type == 'neg':
                new_mask = new_mask[new_mask != 0] - 1 + self._neg_start_idx
            anno_mask[new_origin[1]: new_size[1] + new_origin[1],
                    new_origin[0]: new_size[0] + new_origin[0]] = new_mask
        return selected_mask, anno_mask

    def _generate_img_bg_mask(self):
        self._max_mask_level = self._get_level(self._max_mask_size)
        self._max_mask = np.zeros((self._max_mask_size[1], self._max_mask_size[0]), np.uint8)
        th_img = self._img.read_region((0, 0), self._max_mask_level,
                                       self._img.level_dimensions[self._max_mask_level])
        th_img = th_img.resize(self._max_mask_size)
        # th_mask = self._threshold_downsample_level(th_img)
        th_mask = self._seg_dfs(th_img)
        th_img.close()

        return th_mask

    def _generate_mask(self):

        # init mask without background
        self._min_mask = None
        self._min_mask_size = np.ceil(np.array(self._img.level_dimensions[0])/self._cfg.min_frac).astype(np.int)
        self._min_mask_size = (int(np.ceil(self._img.level_dimensions[0][0] / self._cfg.min_frac)),
                               int(np.ceil(self._img.level_dimensions[0][1] / self._cfg.min_frac)))
        self._min_mask_level = self._get_level(self._min_mask_size)

        th_mask = self._generate_img_bg_mask()

        # Image.fromarray(th_mask * 255).show()
        # th_img.save(os.path.join(self._cfg.vis_ov_mask_folder, os.path.basename(
        #         self._file_name)[:-4] + '.png'))

        assert (self._max_mask_size[1], self._max_mask_size[0]) == th_mask.shape

        if self._mask_files is not None:
            selected_mask, anno_mask = self._merge_mask_files()
            normal_and = np.logical_and(th_mask, selected_mask)

            self._max_mask[selected_mask !=0] = SELECTED
            # self._max_mask[normal_and != 0] = NORMAL

            self._max_mask[anno_mask != 0] = anno_mask[anno_mask!=0]
        else:
            self._max_mask[th_mask != 0] = self._neg_start_idx

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

        mask_pos_idx = np.logical_and(mask >= self._pos_start_idx, mask < self._cfg.num_pos_classes)
        mask_neg_idx = np.logical_and(mask >= self._neg_start_idx, mask < self._cfg.num_neg_classes)
        # pos
        if mask_pos_idx.any():
            img_mask[mask_pos_idx] = self._cfg.alpha * img_np[mask_pos_idx] + \
                                      (1 - self._cfg.alpha) * np.array(TUMOR_COLOR)
        # neg
        if mask_neg_idx.any():
            img_mask[mask_neg_idx] = self._cfg.alpha * img_np[mask_neg_idx] + \
                                       (1 - self._cfg.alpha) * np.array(NORMAL_COLOR)
        if (mask == SELECTED).any():
            img_mask[mask == SELECTED] = self._cfg.alpha * img_np[mask == SELECTED] + \
                                         (1 - self._cfg.alpha) * np.array(SELECTED_COLOR)
        if self._patch_type=='pos':
            if (mask == SAMPLED).any():
                img_mask[mask == SAMPLED] = self._cfg.alpha * img_np[mask == SAMPLED] + \
                                             (1 - self._cfg.alpha) * np.array(TUMOR_COLOR)
        elif self._patch_type == 'neg':
            if (mask == SAMPLED).any():
                img_mask[mask == SAMPLED] = self._cfg.alpha * img_np[mask == SAMPLED] + \
                                            (1 - self._cfg.alpha) * np.array(NORMAL_COLOR)
        else:
            print('patch type error!')
            sys.exit(-1)

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

        # cnt = 0
        if patches['pos'] == []:
            patches = patches['neg']
        else:
            patches = patches['pos']
        random.shuffle(patches)
        for patch in patches:
            # if cnt >= self._cfg.patch_num_in_train:
            #     break
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
            # cnt +=1

    def _get_sampled_patch_mask(self, patches_all):
        lvl = self._get_level((40000, 40000)) + 1
        size = self._img.level_dimensions[lvl]
        sampled_mask = np.zeros((size[1], size[0]), np.uint8)
        frac = size[0]*1.0/self._img.level_dimensions[0][0]
        min_patch_size = int(self._cfg.patch_size*frac)
        patches = []
        if self._patch_type == 'pos':
            if isinstance(patches_all['pos'][0], list):
                patches = patches_all['pos']
            else:
                for p in patches_all['pos']:
                    patches.extend(p)
        else:
            if isinstance(patches_all['pos'][0], list):
                patches = patches_all['neg']
            else:
                for p in patches_all['neg']:
                    patches.extend(p)
        for coor in patches:
            min_coor = (int(coor[0]*frac), int(coor[1]*frac))
            sampled_mask[min_coor[1]: min_coor[1]+min_patch_size,
                min_coor[0]: min_coor[0]+min_patch_size] = SAMPLED
        sampled_mask = np.asarray(Image.fromarray(sampled_mask).resize(self._min_mask_size))
        return sampled_mask

    # test the col raw is right
    def _get_test_mask(self, patches):
        lvl = self._get_level((40000, 40000)) + 1
        size = self._img.level_dimensions[lvl]
        sampled_mask = np.zeros((size[1], size[0]), np.uint8)
        frac = size[0]*1.0/self._img.level_dimensions[0][0]
        min_patch_size = int(self._cfg.patch_size*frac)
        for coor in patches:
            min_coor = (int(coor[0]*frac), int(coor[1]*frac))
            sampled_mask[min_coor[1]: min_coor[1]+min_patch_size,
                min_coor[0]: min_coor[0]+min_patch_size] = SAMPLED
        sampled_mask = np.asarray(Image.fromarray(sampled_mask).resize(self._min_mask_size))
        return sampled_mask

    def _get_train_patch(self):
        do_bg_filter = False
        patches = {'pos': [], 'neg': []}
        for i in range(self._cfg.num_pos_classes):
            patches['pos'].append([])
        for i in range(self._cfg.num_neg_classes):
            patches['neg'].append([])
        assert self._min_mask_size[1], self._min_mask_size[0] == self._min_mask.shape
        num_row, num_col = self._min_mask.shape
        num_row = num_row - self._min_patch_size
        num_col = num_col - self._min_patch_size

        if self._patch_type == 'pos':
            patch_num = self._cfg.pos_patch_num_in_file
        else:
            patch_num = self._cfg.neg_patch_num_in_file

        # step = 1
        row_col = list(product(range(num_row), range(num_col)))
        random.shuffle(row_col)
        cnt = 0
        # ### test raw col
        # tmp_patches = []
        # for row, col in row_col:
        #     tmp_patches.append((int(col * self._cfg.min_frac), int(row * self._cfg.min_frac)))
        # self._get_test_mask(tmp_patches)

        for row, col in row_col:
            if cnt >= patch_num:
                break
            min_patch = self._min_mask[row: row + self._min_patch_size,
                       col: col + self._min_patch_size]
            origin = (int(col * self._cfg.min_frac), int(row * self._cfg.min_frac))

            H, W = min_patch.shape
            H_min = int(np.ceil(H / 8))
            H_max = int(np.ceil(H / 8 * 7))
            W_min = int(np.ceil(W / 8))
            W_max = int(np.ceil(W / 8 * 7))
            # half of the center
            th_num = int(np.ceil((H*3/4 * W*3/4) ))
            if self._patch_type == 'pos':
                for idx in range(self._cfg.num_pos_classes):
                    if np.count_nonzero(min_patch[H_min:H_max, W_min:W_max] == self._pos_start_idx+idx) >= th_num:
                        if do_bg_filter:
                            if self._is_bg(origin):
                                continue
                        patches['pos'][idx].append(origin)
                        self._save_random_patch(origin, min_patch)
                        cnt+=1

            if self._patch_type == 'neg':
                for idx in range(self._cfg.num_neg_classes):
                    # if np.count_nonzero(min_patch[H_min:H_max, W_min:W_max] == NORMAL) >= th_num:
                    if np.count_nonzero(min_patch[H_min:H_max, W_min:W_max] == self._neg_start_idx+idx) > 0:
                        # if do_bg_filter:
                        #     if self._is_bg(origin):
                        #         continue
                        patches['neg'][idx].append(origin)
                        self._save_random_patch(origin, min_patch)
                        cnt+=1
        # visualizaion
        if self._cfg.vis_ov_mask:
            raw_img = self._img.read_region((0, 0), self._min_mask_level,
                                            self._img.level_dimensions[self._min_mask_level])
            raw_img = raw_img.resize(self._min_mask_size)

            mask_np = self._get_sampled_patch_mask(patches)

            sampled_patch_img = self._fusion_mask_img(raw_img, mask_np)

            sampled_patch_img.save(os.path.join(self._cfg.vis_ov_mask_folder, os.path.basename(
                self._file_name)[:-4] + '_sampled_mask' + self._cfg.img_ext))

            sampled_patch_img.close()

        if self._auto_save_patch:
            self._save_patches(patches)
        return patches




def extract(data, file_type, patch_type, auto_save_patch = True):
    img = single_img_process(data['data'][0], data['data'][1], file_type, patch_type, auto_save_patch)
    img._generate_mask()
    return img._get_train_patch()

if __name__ == '__main__':
    pass