import sys
from api import random_divide_data
from api import config_fun
from api import patch_fun
from api import hdf5_fun
import os

cfg = config_fun.config()
if not os.path.exists(cfg.split_file) or cfg.redividing:
    print('not find split file, running random divide data...')
    random_divide_data.random_divide_data()
    print('divide done!')
else:
    print('find the split file, not execute random divide data!')

patch_fun.generate_patch(auto_save_patch=True)

hdf5_fun.convert_patch_to_hdf5()
