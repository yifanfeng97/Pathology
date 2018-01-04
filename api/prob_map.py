from PIL import Image
import numpy as np
import extract_patch_fun
from itertools import product
from torch.utils.data import Dataset
import patch_preprocess_fun
import torch


def _np_resize(data, size):
    return np.asarray(Image.fromarray(data).resize(size))


def _get_input_list(cfg, mask, frac):
    input_list = []
    min_patch_size = np.ceil(cfg.patch_size/frac)
    # is foreground
    def is_fg(patch):
        return np.count_nonzero(patch) > min_patch_size*min_patch_size*2/3
    n_row, n_col = mask.shape
    for row, col in product(range(n_row-min_patch_size), range(n_col-min_patch_size)):
        if is_fg(mask[row: row + min_patch_size, col: col + min_patch_size]):
            # for PIL Image, reverse the row and col
            raw_origin = (np.ceil(col * frac), np.ceil(row * frac))
            out_origin = (row, col)
            input_list.append({'raw': raw_origin, 'out': out_origin})
    return input_list

class gmDataLoader(Dataset):
    def __init__(self, input_list, slide, patch_size):
        # super(testDataLoader, self).__init__()
        self._patch_size = patch_size
        self._data = input_list
        self._slide = slide
        self._compose = patch_preprocess_fun.get_gm_compose()

    def __getitem__(self, index):
        img = self._slide.read_region(self._data[index]['raw'], 0,
                                (self._patch_size, self._patch_size))
        return self._compose(img)

    def __len__(self):
        return len(self._data)


def generate_prob_map(cfg, model, file_name):
    slide = extract_patch_fun.single_img_process(file_name, None, None, None, False)
    img, mask = slide._generate_img_bg_mask()
    size_raw = slide._img.dimensions[0]
    size_out = (np.ceil(size_raw[0]/cfg.gm_stride), np.ceil(size_raw[1]/cfg.gm_stride))

    frac = np.ceil(size_raw[0]/size_out[0])

    img = img.resize(size_out)
    mask = _np_resize(mask, size_out)

    b_map = np.zeros(mask.shape).astype(np.uint8)
    p_map = np.zeros(mask.shape).astype(np.float32)

    input_list = _get_input_list(cfg, mask, frac)

    gm_dataset = gmDataLoader(input_list, slide._img, cfg.patch_size)

    gm_loader = torch.utils.data.DataLoader(gm_dataset, batch_size=cfg.gm_batch_size,
                                            shuffle=False, num_workers=cfg.gm_work_num)







