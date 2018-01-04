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
from itertools import izip

def _precoss_patches(cfg, dataset, file_type):
    data_block = np.zeros((cfg.patch_num_each_hdf5, cfg.patch_size * cfg.patch_size * 3), dtype=np.uint8)
    label_block = np.zeros((cfg.patch_num_each_hdf5, 1), dtype=np.uint8)
    name_block = []
    cnt = 0
    id = 0
    for idx, data in enumerate(tqdm(dataset)):
        img = Image.open(data['file_name'])
        img_np = np.asarray(img).astype(np.uint8)
        img_np = img_np.reshape(-1)
        data_block[cnt] = img_np
        label_block[cnt] = data['label']
        cnt += 1
        name_block.append(os.path.basename(data['file_name']))

        if cnt == cfg.patch_num_each_hdf5 or idx + 1 == len(dataset):
            data_save = data_block[0:cnt]
            label_save = label_block[0:cnt]
            name_save = '\n'.join(name_block)
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
                f.attrs['name'] = name_save
                id += 1
                cnt = 0
                name_block = []


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

def h5_extract_data_label_name(img_size, file_name):
    data = None
    label = None
    name = None
    with h5py.File(file_name) as f:
        data = f['data'].value
        label = f['label'].value
        name = f.attrs.values()
    return data.reshape(-1, img_size, img_size, 3), label, name[0].split('\n')

def get_all_data_label_name(cfg, train):
    file_names = None
    data = None
    label = None
    name = None
    if train:
        file_names = glob.glob(cfg.patch_hdf5_train_file_pre + '*')
    else:
        file_names = glob.glob(cfg.patch_hdf5_val_file_pre + '*')
    for file_name in file_names:
        t_data, t_label, t_name = h5_extract_data_label_name(cfg.patch_size, file_name)
        if data is None:
            data = t_data
            label = t_label
            name = t_name
        else:
            data = np.concatenate((data, t_data), axis=0)
            label = np.concatenate((label, t_label), axis=0)
            name.extend(t_name)
    return data, label, name


class h5_dataloader(Dataset):
    def __init__(self, train = True):
        cfg = config_fun.config()
        # self._raw_size = cfg.patch_size
        self._train = train
        self._compose = patch_preprocess_fun.get_train_val_compose()
        self._data, self._label, self._name = get_all_data_label_name(cfg, train=train)
        assert self._data.shape[0] == self._label.shape[0]

    def __getitem__(self, index):
        return self._compose(self._data[index]), self._label[index]

    def __len__(self):
        return self._label.shape[0]

def _random_vis_hdf5(cfg, train):
    data, label, name = get_all_data_label_name(cfg, train)
    save_dir_pre = cfg.vis_hdf5_folder
    if train:
        file_type = 'train'
    else:
        file_type = 'val'
    save_dir_pre = os.path.join(save_dir_pre, file_type)
    cfg.check_dir(save_dir_pre)
    save_dir_pos = os.path.join(save_dir_pre, 'pos')
    save_dir_neg = os.path.join(save_dir_pre, 'neg')
    cfg.check_dir(save_dir_pos)
    cfg.check_dir(save_dir_neg)
    for idx, n in enumerate(tqdm(name)):
        l = label[idx]
        d = data[idx]
        if random.random() < cfg.vis_hdf5_prob:
            img = Image.fromarray(d)
            if l == 1:
                img.save(os.path.join(save_dir_pos, n))
            else:
                img.save(os.path.join(save_dir_neg, n))
            img.close()

def random_vis_hdf5():
    cfg = config_fun.config()
    print('vis train hdf5 ~')
    _random_vis_hdf5(cfg, train=True)
    print('vis validation hdf5 ~')
    _random_vis_hdf5(cfg, train=False)


if __name__ == '__main__':
    random_vis_hdf5()