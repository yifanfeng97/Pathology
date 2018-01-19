from api import config_fun
from api import prob_map_fcn
from api import prob_map_cls
import numpy as np
import train_helper
import os
import torch
from PIL import Image

cfg = config_fun.config()
model = train_helper.get_model(cfg, load_param_from_folder=True)

model.cuda()

def norm(p):
    return ((p-p.min())/(p.max()-p.min()))

f = open(cfg.test_file, 'r')
for s in f.readlines():
    if s.split()==[]: continue
    file_name, label = s.split('*')
    print('processing ' + file_name)

    raw_img, b_map, p_map = prob_map_fcn.generate_prob_map(cfg, model, file_name)

    save_dir_pre = os.path.join(cfg.gm_foder,
                                os.path.basename(file_name).split('.')[0])
    raw_img_dir = save_dir_pre + '_raw_img' + cfg.img_ext
    b_map_img_dir = save_dir_pre + '_b_map_img' + cfg.img_ext
    p_map_img_dir = save_dir_pre + '_p_map_img' + cfg.img_ext

    b_map_npy_dir = save_dir_pre + '_b_map_img' + '.txt'
    p_map_npy_dir = save_dir_pre + '_p_map_img' + '.txt'

    np.savetxt(b_map_npy_dir, b_map)
    np.savetxt(p_map_npy_dir, p_map)

    raw_img.save(raw_img_dir)
    Image.fromarray((p_map*255).astype(np.uint8)).save(p_map_img_dir)
    Image.fromarray(b_map*255).save(b_map_img_dir)

    raw_img.close()


