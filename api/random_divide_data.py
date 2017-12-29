import sys
# sys.path.append('../')
import glob
import os
import slide_fun
import openslide
import random
import numpy as np
import json

img_type = ['*.svs', '*.kfb']

from config_fun import config

def _remove_corrupted_files(files):
    corrent_files = []
    for item in files:
        data_file = item[0]
        try:
            data = slide_fun.AllSlide(data_file)
            level = data.level_count -1
            level_size = data.level_dimensions[level]
            while 0 in level_size:
                level -= 1
                level_size = data.level_dimensions[level]
            patch = data.read_region((0, 0), level, level_size)
            patch.close()
            corrent_files.append(item)
        except openslide.OpenSlideUnsupportedFormatError:
            print(data_file, 'unsupported error')
        except openslide.lowlevel.OpenSlideError:
            print(data_file, 'low level error')
    return corrent_files

def _split_dataset(files, info, test_frac, val_num):
    data_num = len(files)
    test_num = int(np.floor(test_frac * data_num))
    data_idx = range(data_num)
    random.shuffle(data_idx)

    train_files = []
    test_files = []
    val_files = []
    # test
    for idx in data_idx[:test_num]:
        test_files.append({'data': files[idx], 'info': 'test_'+info})
    # train
    for idx in data_idx[test_num: data_num - val_num]:
        train_files.append({'data': files[idx], 'info': 'train_'+info})
    # validation
    for idx in data_idx[data_num - val_num:]:
        val_files.append({'data': files[idx], 'info': 'val_'+info})
    return train_files, val_files, test_files

def random_divide_data():
    # get configuration file
    cfg = config()

    tumor_anno_files = glob.glob(os.path.join(cfg.tumor_anno_folder,
                                              '*.mask'))

    tumor_data_files = []
    normal_data_files = []
    for t in img_type:
        tumor_data_files.extend(glob.glob(os.path.join(cfg.tumor_data_folder, t)))
        normal_data_files.extend(glob.glob(os.path.join(cfg.normal_data_folder, t)))

    print('find %d tumor files, %d tumor annotation files, %d normal files'%
          (len(tumor_data_files), len(tumor_anno_files), len(normal_data_files)))

    tumor_data_to_masks = {}
    for anno_file in tumor_anno_files:
        data_file = slide_fun.get_mask_info(os.path.basename(anno_file))[0]
        if data_file not in tumor_data_to_masks:
            tumor_data_to_masks[data_file] = [anno_file]
        else:
            tumor_data_to_masks[data_file].append(anno_file)
    print('find annotation of %d tumor files'%len(tumor_data_to_masks))

    tumor_data_to_path = {}
    for file_path in tumor_data_files:
        data_file = os.path.basename(file_path).split('.')[0]
        tumor_data_to_path[data_file] = file_path

    tumor_data = []
    normal_data = []
    for data_file, mask_files in tumor_data_to_masks.iteritems():
        if data_file in tumor_data_to_path:
            tumor_data.append((tumor_data_to_path[data_file], mask_files))

    for data_file in normal_data_files:
        normal_data.append((data_file, None))

    tumor_data = _remove_corrupted_files(tumor_data)
    normal_data = _remove_corrupted_files(normal_data)

    print('finally\ntumor files %d'%len(tumor_data))
    print('normal files %d'%len(normal_data))

    # split data
    split_data = []
    # tumor
    train_tumor_files, val_tumor_files, test_tumor_files = _split_dataset(
                tumor_data, 'tumor', cfg.test_frac, cfg.val_tumor)
    split_data.extend(train_tumor_files)
    split_data.extend(val_tumor_files)
    split_data.extend(test_tumor_files)
    # normal
    train_normal_files, val_normal_files, test_normal_files = _split_dataset(
                normal_data, 'normal', cfg.test_frac, cfg.val_normal)
    split_data.extend(train_normal_files)
    split_data.extend(val_normal_files)
    split_data.extend(test_normal_files)

    print('train tumor %d, normal %d'%
          (len(train_tumor_files), len(train_normal_files)))
    print('val tumor %d, normal %d'%
          (len(val_tumor_files), len(val_normal_files)))
    print('test tumor %d, normal %d'%
          (len(test_tumor_files), len(test_normal_files)))

    with open(cfg.split_file, 'w') as f:
        json.dump(split_data, f)

    print('All split files information will be written into' + cfg.split_file)

    with open(cfg.test_file, 'w') as f:
        for item in test_tumor_files:
            f.write(item['data'][0]+ ' 1' + '\n')
        for item in test_normal_files:
            f.write(item['data'][0]+ ' 0' + '\n')

if __name__ == '__main__':
    random_divide_data()


