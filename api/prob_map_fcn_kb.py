from PIL import Image
import numpy as np
import extract_patch_fun
from itertools import product
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
from torch.autograd import Variable
from api.meter import timemeter
import dataloader_fun
import sys


def _np_resize(data, size):
    return np.asarray(Image.fromarray(data).resize(size))


# is foreground
def is_fg(patch, min_patch_size):
    # return np.count_nonzero(patch) > 0
    return np.count_nonzero(patch) > min_patch_size*min_patch_size*1.0/2
    # return np.count_nonzero(patch) > min_patch_size*min_patch_size*9/10

def get_coor(row, col, patch_in_size, off_set):
    return (patch_in_size - off_set) * col, (patch_in_size - off_set) * row


def _get_input_list(mask, mask_frac, patch_in_size, size_raw, size_out, patch_out_size):

    input_list = []

    off_set = 192
    num_row = int(np.ceil((size_raw[1] - patch_in_size)*1.0/(patch_in_size - off_set))) #+ 1)
    num_col = int(np.ceil((size_raw[0] - patch_in_size)*1.0/(patch_in_size - off_set))) #+ 1
    min_patch_size = int(patch_in_size/mask_frac)
    cnt_all = 0

    for row in range(num_row):
        for col in range(num_col):
            raw_origin = get_coor(row, col, patch_in_size, off_set)
            mask_origin = (int(raw_origin[1]/mask_frac), int(raw_origin[0]/mask_frac))
            if is_fg(mask[mask_origin[0]: mask_origin[0] + min_patch_size,
                     mask_origin[1]: mask_origin[1] + min_patch_size], min_patch_size):
                input_list.append({'raw':raw_origin, 'out':(row*patch_out_size, col*patch_out_size)})
            cnt_all += 1
    print('input %d/%d (%.3f%%)' % (len(input_list), cnt_all, len(input_list)*1.0/cnt_all*100))
    return input_list


    # stride_row = cal_stride(patch_in_size, patch_out_size, size_raw[1], size_out[1])
    # stride_col = cal_stride(patch_in_size, patch_out_size, size_raw[0], size_out[0])
    # min_stride_row = int(stride_row/mask_frac)
    # min_stride_col = int(stride_col/mask_frac)
    # min_patch_size = int(patch_in_size/mask_frac)
    # cnt_all = 0
    #
    # for row in range(0, mask.shape[0]-min_stride_row, min_stride_row):
    #     for col in range(0, mask.shape[1]-min_stride_col, min_stride_col):
    #         if is_fg(mask[row: row + min_patch_size, col: col + min_patch_size], min_patch_size):
    #             raw_origin = (int(col* mask_frac), int(row*mask_frac))
    #             out_origin = ((row/min_stride_row) * patch_out_size, (col/min_stride_col) * patch_out_size)
    #             # out_origin = (row, col)
    #             input_list.append({'raw': raw_origin, 'out': out_origin})
    #         cnt_all += 1
    # print('input %d/%d (%.3f%%)' % (len(input_list), cnt_all, len(input_list)*1.0/cnt_all*100))
    # return input_list


def _get_label_prob(data_loader, model):
    output = None
    softmax = torch.nn.Softmax(dim=1)
    for i, inputs_img in enumerate(tqdm(data_loader)):
        inputs_img = Variable(inputs_img).cuda()
        preds = model(inputs_img)
        preds = softmax(preds)
        if output is None:
            output = preds.data.cpu().numpy()
        else:
            output = np.concatenate((output, preds.data.cpu().numpy()), axis=0)
    return output


def _fill_list_into_map(input_list, maps, output, patch_out_size):
    for idx, item in enumerate(tqdm(input_list)):
        maps[item['out'][0]: item['out'][0]+patch_out_size,
                item['out'][1]:item['out'][1]+patch_out_size] = output[idx]#.transpose()
    return maps


def cal_new_len(img_size):
    if img_size<=192:
        print('img_size is smaller than 192')
        sys.exit(-1)
    # return int((img_size-192)/(windows_size-192))
    return int((img_size/32) - 6)


def cal_stride(windows_size, windows_out_size, img_size, img_out_size):
    return int((img_size-windows_size)*1.0/(img_out_size - windows_out_size)*windows_out_size)


def generate_prob_map(cfg, model, file_name):
    t = timemeter.TimeMeter()
    slide = extract_patch_fun.single_img_process(file_name, None, None, None, False)
    print('start extract background ')
    img, mask = slide._generate_img_bg_mask()
    print('Done! %.4fs'%t.value())

    size_raw = slide._img.level_dimensions[0]

    mask_frac = size_raw[1]*1.0/mask.shape[0]

    # calculate the gm row column and patch out size
    row = cal_new_len(size_raw[1])
    col = cal_new_len(size_raw[0])
    patch_out_size = cal_new_len(cfg.windows_size)

    size_out = (col, row)

    img = img.resize(size_out)
    # mask = _np_resize(mask, (size_out[1], size_out[0]))

    b_map = np.zeros((row, col)).astype(np.uint8)
    p_map = np.zeros((row, col)).astype(np.float32)

    print('start get input list ')
    input_list = _get_input_list(mask, mask_frac, cfg.windows_size, size_raw, size_out, patch_out_size)
    print('Done! %.4fs' % t.value())
    print('get %d input patch'%len(input_list))

    gm_dataset = dataloader_fun.gm_fcn_DataLoader(input_list, slide._img, cfg.windows_size)

    gm_loader = torch.utils.data.DataLoader(gm_dataset, batch_size=cfg.gm_batch_size,
                                            shuffle=False, num_workers=cfg.gm_work_num)
    print('start inference the model')
    output = _get_label_prob(gm_loader, model)
    print('Done! %.4fs' % t.value())
    output_b = np.argmax(output, axis=1)
    output_p = output[:, 1]

    print('start fill the output into the map')
    b_map = _fill_list_into_map(input_list, b_map, output_b, patch_out_size)
    p_map = _fill_list_into_map(input_list, p_map, output_p, patch_out_size)
    print('Done! %.4fs' % t.value())

    return img, b_map, p_map






