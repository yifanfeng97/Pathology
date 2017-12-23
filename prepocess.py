import sys
from api import random_divide_data
from api import get_config
import os

cfg = get_config.config()
if not os.path.exists(cfg.split_file) or cfg.redividing:
    print('not find split file, running random divide data...')
    random_divide_data.random_divide_data()
    print('divide done!')
else:
    print('find the split file, not execute random divide data!')
