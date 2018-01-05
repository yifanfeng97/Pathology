from PIL import Image
import numpy as np
import extract_patch_fun
from itertools import product
from torch.utils.data import Dataset
import patch_preprocess_fun
import torch
from tqdm import tqdm
from torch.autograd import Variable
from api.meter import timemeter


def _np_resize(data, size):
    return np.asarray(Image.fromarray(data).resize(size))


def _get_input_list(cfg, mask, frac):
    input_list = []
    min_patch_size = int(np.ceil(cfg.patch_size/frac))
    # is foreground
    def is_fg(patch):
        return np.count_nonzero(patch) > min_patch_size*min_patch_size*2/3
    n_row, n_col = mask.shape
    for row, col in product(range(n_row-min_patch_size), range(n_col-min_patch_size)):
        if is_fg(mask[row: row + min_patch_size, col: col + min_patch_size]):
            # for PIL Image, reverse the row and col
            raw_origin = (int(np.ceil(col * frac)), int(np.ceil(row * frac)))
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


def _get_label_prob(data_loader, model):
    output = None
    # model.cuda()
    for i, inputs_img in enumerate(tqdm(data_loader)):
        inputs_img = Variable(inputs_img).cuda()
        preds = model(inputs_img)
        if output is None:
            output = preds.data.cpu().squeeze().numpy()
        else:
            output = np.vstack((output, preds.data.cpu().squeeze().numpy()))
    return output


def _fill_list_into_map(input_list, maps, output):
    for idx, item in enumerate(tqdm(input_list)):
        maps[item['out']] = output[idx]
    return maps

def generate_prob_map(cfg, model, file_name):
    t = timemeter.TimeMeter()
    slide = extract_patch_fun.single_img_process(file_name, None, None, None, False)
    print('start extract background ')
    img, mask = slide._generate_img_bg_mask()
    print('Done! %.4fs'%t.value())
    size_raw = slide._img.level_dimensions[0]
    size_out = (int(np.ceil(size_raw[0]/cfg.gm_stride)), int(np.ceil(size_raw[1]/cfg.gm_stride)))

    frac = np.ceil(size_raw[0]/size_out[0])

    img = img.resize(size_out)
    mask = _np_resize(mask, size_out)

    b_map = np.zeros(mask.shape).astype(np.uint8)
    p_map = np.zeros(mask.shape).astype(np.float32)

    print('start get input list ')
    input_list = _get_input_list(cfg, mask, frac)
    print('Done! %.4fs' % t.value())
    print('get %d input patch'%len(input_list))

    gm_dataset = gmDataLoader(input_list, slide._img, cfg.patch_size)

    gm_loader = torch.utils.data.DataLoader(gm_dataset, batch_size=cfg.gm_batch_size,
                                            shuffle=False, num_workers=cfg.gm_work_num)
    print('start inference the model')
    output = _get_label_prob(gm_loader, model)
    print('Done! %.4fs' % t.value())
    output_b = np.argmax(output, axis=1)
    output_p = output[:, 1]

    print('start fill the output into the map')
    b_map = _fill_list_into_map(input_list, b_map, output_b)
    p_map = _fill_list_into_map(input_list, p_map, output_p)
    print('Done! %.4fs' % t.value())

    return img, b_map, p_map






