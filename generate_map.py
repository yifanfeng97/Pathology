from api import config_fun
# from api import prob_map_fcn as prob_map
from api import prob_map_fcn_kb as prob_map
# from api import prob_map_cls as prob_map
from api import heat_map_fun
from api import slide_fun
import openslide
import numpy as np
import train_helper
import os
import torch
from PIL import Image
# import matplotlib.pyplot as plt

cfg = config_fun.config()
# torch.cuda.set_device(0)
model = train_helper.get_model(cfg, load_param_from_folder=True)

model.cuda()
model.eval()

f = open(cfg.test_file, 'r')
for s in f.readlines():
    if s.split() == []: continue
    # file_name, label = s.split('*')
    file_name = s.split('\n')[0]

    if os.path.exists(file_name + '.heatmap.jpg'):
        continue

    try:
        data = slide_fun.AllSlide(file_name)
        level = data.level_count - 1
        level_size = data.level_dimensions[level]
        while 0 in level_size:
            level -= 1
            level_size = data.level_dimensions[level]
        patch = data.read_region((0, 0), level, level_size)
        patch.close()
    except openslide.OpenSlideUnsupportedFormatError:
        # print(file_name, 'unsupported error')
        continue
    except openslide.lowlevel.OpenSlideError:
        # print(file_name, 'low level error')
        continue
    print('processing ' + file_name)

    raw_img, b_map, p_map = prob_map.generate_prob_map(cfg, model, file_name)
    save_dir_pre = os.path.join(cfg.gm_foder,
                                os.path.basename(file_name).split('.')[0])
    raw_img_dir = save_dir_pre + '_raw_img' + cfg.img_ext
    h_map_img_dir = save_dir_pre + '_h_map_img' + cfg.img_ext
    p_map_npy_dir = save_dir_pre + '_p_map_img' + '.npy'

    np.save(p_map_npy_dir, p_map)
    np.save(file_name+'.pmap.npy', p_map)

    raw_img.save(raw_img_dir)
    raw_img.save(file_name + '.raw.jpg')
    raw_img.close()
    htmap_img = heat_map_fun.get_heatmap_from_prob(p_map)
    htmap_img.save(h_map_img_dir)
    htmap_img.save(file_name + '.heatmap.jpg')
    htmap_img.close()



