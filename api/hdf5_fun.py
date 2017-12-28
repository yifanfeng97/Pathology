import h5py
import numpy as np
import config_fun
import glob
import os
import random
from PIL import Image

def _precoss_patches(cfg, dataset, file_type):
    data_block = np.zeros((cfg.patch_num_each_hdf5, cfg.patch_size * cfg.patch_size), dtype=np.uint8)
    label_block = np.zeros((cfg.patch_num_each_hdf5, 1), dtype=np.uint8)
    cnt = 0
    id = 0
    for data in dataset:
        img = Image.open(data['file_name'])
        img = np.asarray(img).astype(np.uint8)
        img = img.reshape(-1)
        data_block[cnt] = img
        label_block[cnt] = data['label']
        cnt +=1

        if cnt==cfg.patch_num_each_hdf5:
            prefix = ''
            if file_type == 'train':
                prefix = cfg.patch_hdf5_train_file_pre
            else:
                prefix = cfg.patch_hdf5_val_file_pre
            with h5py.File('%s_%d.h5'%(prefix, id), 'w') as f:
                f.create_dataset('data', shape=data_block.shape, compression = 'gzip', dtype='uint8')
                f.create_dataset('label', shape=label_block.shape, compression = 'gzip', dtype='uint8')
                f['data'] = data_block
                f['label'] = label_block

                id += 1
                cnt = 0






def _add_label(pos_patches, neg_patches):
    patches = []
    for patch in pos_patches:
        patches.append({'file_name': patch, 'label': 1})
    for patch in neg_patches:
        patches.append({'file_name': patch, 'label': 0})
    random.shuffle(patches)
    return patches

def convert_patch_to_hdf5():
    cfg = config_fun.config()
    train_pos_patches = glob.glob(os.path.join(cfg.patch_save_folder, 'train', 'pos', '*'+ cfg.img_ext))
    train_neg_patches = glob.glob(os.path.join(cfg.patch_save_folder, 'train', 'neg', '*'+ cfg.img_ext))
    val_pos_patches = glob.glob(os.path.join(cfg.patch_save_folder, 'val', 'pos', '*'+ cfg.img_ext))
    val_neg_patches = glob.glob(os.path.join(cfg.patch_save_folder, 'val', 'neg', '*'+ cfg.img_ext))

    # f = h5py.File(cfg.patch_hdf5_file)
    train_patches = _add_label(train_pos_patches, train_neg_patches)
    val_patches = _add_label(val_pos_patches, val_neg_patches)
    _precoss_patches(cfg, train_patches, 'train')
    _precoss_patches(cfg, val_patches, 'val')
