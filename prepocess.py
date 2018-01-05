import sys
from api import random_divide_data
from api import config_fun
from api import patch_fun
from api import hdf5_fun
import os
import train

def main():
    cfg = config_fun.config()
    if not os.path.exists(cfg.split_file) or cfg.redividing:
        print('not find split file, running random divide data...')
        random_divide_data.random_divide_data()
        print('divide done!')
    else:
        print('find the split file, not execute random divide data!')

    if cfg.redividing or cfg.regenerate:
        print('generate patch~')
        patch_fun.generate_patch(auto_save_patch=True)
    else:
        print('not generate patch~')
    print('convert patches to hdf5 files~')
    hdf5_fun.convert_patch_to_hdf5()
    # vis the packaged hdf5 file
    hdf5_fun.random_vis_hdf5()

if __name__ == '__main__':
    main()
