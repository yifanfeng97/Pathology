from api import config_fun
from api import prob_map_fcn as prob_map
# from api import prob_map_cls as prob_map
from api import heat_map_fun
import numpy as np
import train_helper
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt

cfg = config_fun.config()
# torch.cuda.set_device(0)
model = train_helper.get_model(cfg, load_param_from_folder=True)

model.cuda()


f = open(cfg.test_file, 'r')
for s in f.readlines():
    if s.split() == []: continue
    # file_name, label = s.split('*')
    file_name = s.split('\n')[0]
    print('processing ' + file_name)

    raw_img, b_map, p_map = prob_map.generate_prob_map(cfg, model, file_name)

    save_dir_pre = os.path.join(cfg.gm_foder,
                                os.path.basename(file_name).split('.')[0])
    raw_img_dir = save_dir_pre + '_raw_img' + cfg.img_ext
    b_map_img_dir = save_dir_pre + '_b_map_img' + cfg.img_ext
    p_map_img_dir = save_dir_pre + '_p_map_img' + cfg.img_ext
    h_map_img_dir = save_dir_pre + '_h_map_img' + cfg.img_ext

    b_map_npy_dir = save_dir_pre + '_b_map_img' + '.npy'
    p_map_npy_dir = save_dir_pre + '_p_map_img' + '.npy'

    np.save(b_map_npy_dir, b_map)
    np.save(p_map_npy_dir, p_map)

    raw_img.save(raw_img_dir)
    raw_img.close()
    Image.fromarray((p_map*255).astype(np.uint8)).save(p_map_img_dir)
    Image.fromarray(b_map*255).save(b_map_img_dir)
    heat_map_fun.get_heatmap_from_prob(prob_map).save(h_map_img_dir)



