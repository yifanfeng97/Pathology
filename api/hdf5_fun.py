import h5py
import numpy as np
import config_fun
import glob
import os
import random
import sys
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
import patch_preprocess_fun

def _precoss_patches(cfg, dataset, file_type):
    data_block = np.zeros((cfg.patch_num_each_hdf5, cfg.patch_size * cfg.patch_size * 3), dtype=np.uint8)
    label_block = np.zeros((cfg.patch_num_each_hdf5, 1), dtype=np.uint8)
    cnt = 0
    id = 0
    for idx, data in enumerate(tqdm(dataset)):
        img = Image.open(data['file_name'])
        img_np = np.asarray(img).astype(np.uint8)
        img_np = img_np.reshape(-1)
        data_block[cnt] = img_np
        label_block[cnt] = data['label']
        cnt += 1

        if cnt == cfg.patch_num_each_hdf5 or idx + 1 == len(dataset):
            data_save = data_block[0:cnt]
            label_save = label_block[0:cnt]
            prefix = ''
            if file_type == 'train':
                prefix = cfg.patch_hdf5_train_file_pre
            else:
                prefix = cfg.patch_hdf5_val_file_pre
            with h5py.File('%s_%d.h5'%(prefix, id), 'w') as f:
                f.create_dataset('data', shape=data_save.shape,
                                 compression='gzip', dtype='uint8', data=data_save)
                f.create_dataset('label', shape=label_save.shape,
                                 compression='gzip', dtype='uint8', data=label_save)
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

    train_patches = _add_label(train_pos_patches, train_neg_patches)
    val_patches = _add_label(val_pos_patches, val_neg_patches)
    print('processing train patches~')
    _precoss_patches(cfg, train_patches, 'train')
    print('processing validation patches~')
    _precoss_patches(cfg, val_patches, 'val')

def h5_extract_data_label(img_size, file_name):
    data = None
    label = None
    with h5py.File(file_name) as f:
        data = f['data'].value
        label = f['label'].value
    return data.reshape(-1, img_size, img_size, 3), label

class h5_dataloader(Dataset):
    def __init__(self, train = True):
        cfg = config_fun.config()
        self._raw_size = cfg.patch_size
        self._train = train
        self._compose = patch_preprocess_fun.get_compose()
        file_names = []
        if self._train:
            file_names = glob.glob(cfg.patch_hdf5_train_file_pre + '*')
        else:
            file_names = glob.glob(cfg.patch_hdf5_val_file_pre + '*')

        self._data = None
        self._label = None
        for file_name in file_names:
            if self._data is None:
                self._data, self._label = h5_extract_data_label(self._raw_size, file_name)
            else:
                t_data, t_label = h5_extract_data_label(self._raw_size, file_name)
                self._data = np.concatenate((self._data, t_data), axis=0)
                self._label = np.concatenate((self._label,t_label), axis=0)
                del t_data, t_label
        assert self._data.shape[0] == self._label.shape[0]

    def __getitem__(self, index):
        return self._compose(self._data[index]), self._label[index]

    def __len__(self):
        return self._label.shape[0]


if __name__ == '__main__':
    pass