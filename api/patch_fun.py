import slide_fun
import config_fun
import json
import extract_patch_fun
import os
from tqdm import tqdm
import numpy as np
import glob

def _prepare_data(cfg, data, file_type, auto_save_patch = True):
    # patches = []
    for idx, item in enumerate(tqdm(data)):
        filename = item['data'][0]
        coor_file_name = os.path.join(cfg.patch_coor_folder,
                    'coor' + os.path.basename(filename).split('.')[0] + '.npy')
        if os.path.exists(coor_file_name):
            print('find ' + coor_file_name)
            continue

        print('processing img: ' + filename)
        if 'tumor' in item['info']:
            patch_type = 'pos'
        else:
            patch_type = 'neg'
        patch = extract_patch_fun.extract(item, file_type, patch_type, auto_save_patch = auto_save_patch)
        patch_cell = {'data': filename,
                        'info': item['info'],
                        'patch': patch}
        # patches.append(patch_cell)
        print('get patches from %s, pos:%d, neg:%d\n'%
              (os.path.basename(filename), len(patch['pos']), len(patch['neg'])))
        print('save patch coor into file ', coor_file_name)
        np.save(coor_file_name, patch_cell)
    # return patches


def get_coor(cfg, file_type):
    patches = []
    file_names = glob.glob(os.path.join(cfg.patch_coor_folder, '*'))
    for file_name in file_names:
        coor = np.load(file_name)
        coor = coor.item()
        if coor['info'].startswith(file_type):
            patches.append(coor)
    return patches

def generate_patch(auto_save_patch = True):
    cfg = config_fun.config()
    with open(cfg.split_file) as f:
        split_data = json.load(f)

    train_data = filter(lambda item: item['info'] == 'train_tumor' or
                                    item['info'] == 'train_normal', split_data)
    val_data   = filter(lambda item: item['info'] == 'val_tumor' or
                                    item['info'] == 'val_normal', split_data)
    # test_data  = filter(lambda item: item['info'] == 'test_tumor' or
    #                                 item['info'] == 'test_normal', split_data)

    _prepare_data(cfg, train_data, 'train', auto_save_patch = auto_save_patch)
    _prepare_data(cfg, val_data, 'val', auto_save_patch = auto_save_patch)

    train_patch = get_coor(cfg, 'train')
    val_patch = get_coor(cfg, 'val')

    print('train file %d'%len(train_patch))
    print('val file %d'%len(val_patch))
    # np.save(cfg.patch_coor_file, train_patch + val_patch)
    # print('save all patches into file %s'%cfg.patch_coor_file)
