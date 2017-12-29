import slide_fun
import config_fun
import json
import extract_patch_fun
import os
from tqdm import tqdm
import numpy as np

def _prepare_data(data, file_type, auto_save_patch = True):
    patches = []
    for idx, item in enumerate(tqdm(data)):
        print('processing img: ' + item['data'][0])
        patch = []
        if 'tumor' in item['info']:
            patch = extract_patch_fun.extract(item, file_type, 'pos', auto_save_patch = auto_save_patch)
        else:
            patch = extract_patch_fun.extract(item, file_type, 'neg', auto_save_patch = auto_save_patch)
        patches.append({'data': item['data'][0],
                        'info': item['info'], 'patch':patch})
        print('get patches from %s, pos:%d, neg:%d\n'%
              (os.path.basename(item['data'][0]), len(patch['pos']), len(patch['neg'])))
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

    train_patch = _prepare_data(train_data, 'train', auto_save_patch = auto_save_patch)
    val_patch = _prepare_data(val_data, 'val', auto_save_patch = auto_save_patch)

    np.save(cfg.patch_coor_file, train_patch + val_patch)
    print('save all patches into file %s'%cfg.patch_coor_file)
