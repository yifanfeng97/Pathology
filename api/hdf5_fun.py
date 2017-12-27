import h5py
import numpy as np
import config_fun
import glob
import os

def convert_patch_to_hdf5():
    cfg = config_fun.config()
    train_pos_patches = glob.glob(os.path.join(cfg.patch_save_folder, 'train', 'pos', '*'+ cfg.img_ext))
    train_neg_patches = glob.glob(os.path.join(cfg.patch_save_folder, 'train', 'neg', '*'+ cfg.img_ext))
    val_pos_patches = glob.glob(os.path.join(cfg.patch_save_folder, 'val', 'pos', '*'+ cfg.img_ext))
    val_neg_patches = glob.glob(os.path.join(cfg.patch_save_folder, 'val', 'neg', '*'+ cfg.img_ext))