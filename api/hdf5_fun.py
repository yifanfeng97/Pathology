import h5py
import numpy as np
import config_fun
import glob
import os
import random
import sys
from PIL import Image
from torch.utils.data import Dataset

def _precoss_patches(cfg, dataset, file_type):
    data_block = np.zeros((cfg.patch_num_each_hdf5, cfg.patch_size * cfg.patch_size), dtype=np.uint8)
    label_block = np.zeros((cfg.patch_num_each_hdf5, 1), dtype=np.uint8)
    cnt = 0
    id = 0
    for data in dataset:
        img = Image.open(data['file_name'])
        img_np = np.asarray(img).astype(np.uint8)
        img_np = img_np.reshape(-1)
        data_block[cnt] = img_np
        label_block[cnt] = data['label']
        cnt +=1

        if cnt == cfg.patch_num_each_hdf5:
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

    train_patches = _add_label(train_pos_patches, train_neg_patches)
    val_patches = _add_label(val_pos_patches, val_neg_patches)
    _precoss_patches(cfg, train_patches, 'train')
    _precoss_patches(cfg, val_patches, 'val')

def h5_extract_data_label(img_size, file_name):
    data = None
    label = None
    with h5py.File(file_name) as f:
        data = f['data']
        label = f['label']
    return data.reshape(-1, img_size, img_size), label

class h5_dataloader(Dataset):
    def __init__(self, input_size = 224, train = True):
        cfg = config_fun.config()
        self._raw_size = cfg.patch_size
        self._input_size = input_size
        self._train = train
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
                self._data.vstack(t_data)
                self._label.vstack(t_label)
                del t_data, t_label
        assert self._data.size(0) == self._label.size(0)

    def __getitem__(self, index):
        return self._data[index], self._label[index]

    def __len__(self):
        return self._label.size(0)

def get_h5_dataloader(cfg, train):
    if cfg.model == 'vgg':
        return h5_dataloader(224, train)
    elif cfg.model == 'googlenet':
        return h5_dataloader(299, train)
    elif cfg.model == 'resnet':
        return h5_dataloader(224, train)
    print('not support the model: '+ cfg.model)
    sys.exit(-1)

if __name__ == '__main__':
    pass