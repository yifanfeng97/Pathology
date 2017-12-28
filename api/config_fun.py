import sys
import os
# sys.path.append('../')
import ConfigParser

class config():
    def __init__(self, cfg_file = 'config/path.cfg'):

        cfg = ConfigParser.SafeConfigParser()
        cfg.read(cfg_file)
        # read content
        ## default
        ### folder
        self.normal_data_folder = cfg.get('DEFAULT', 'normal_data_folder')
        self.tumor_data_folder = cfg.get('DEFAULT', 'tumor_data_folder')
        self.tumor_anno_folder = cfg.get('DEFAULT', 'tumor_anno_folder')
        self.result_folder = cfg.get('DEFAULT', 'result_folder')
        ### others
        self.img_ext = cfg.get('DEFAULT', 'img_ext')
        ## PREPROCESS
        ### folder
        self.split_folder = cfg.get('PREPROCESS', 'split_folder')
        self.patch_coor_folder = cfg.get('PREPROCESS', 'patch_coor_folder')
        self.vis_ov_mask_folder = cfg.get('PREPROCESS', 'vis_ov_mask_folder')
        self.vis_patch_folder = cfg.get('PREPROCESS', 'vis_patch_folder')
        self.vis_pos_patch_folder = cfg.get('PREPROCESS', 'vis_pos_patch_folder')
        self.vis_neg_patch_folder = cfg.get('PREPROCESS', 'vis_neg_patch_folder')
        self.patch_save_folder = cfg.get('PREPROCESS', 'patch_save_folder')

        ### number
        self.val_normal = cfg.getint('PREPROCESS', 'val_normal')
        self.val_tumor = cfg.getint('PREPROCESS', 'val_tumor')
        self.patch_size = cfg.getint('PREPROCESS', 'patch_size')
        self.patch_num_in_file = cfg.getint('PREPROCESS', 'patch_num_in_file')
        self.patch_num_in_train = cfg.getint('PREPROCESS', 'patch_num_in_train')
        self.test_frac = cfg.getfloat('PREPROCESS', 'test_frac')
        self.alpha = cfg.getfloat('PREPROCESS', 'alpha')
        self.max_frac = cfg.getfloat('PREPROCESS', 'max_frac')
        self.min_frac = cfg.getfloat('PREPROCESS', 'min_frac')
        self.vis_patch_prob = cfg.getfloat('PREPROCESS', 'vis_patch_prob')

        ### files
        self.split_file = cfg.get('PREPROCESS', 'split_file')
        self.test_file = cfg.get('PREPROCESS', 'test_file')
        ### others
        self.redividing = cfg.getboolean('PREPROCESS', 'redividing')
        self.vis_ov_mask = cfg.getboolean('PREPROCESS', 'vis_ov_mask')

        ## TRAIN
        ### folder
        self.train_folder = cfg.get('TRAIN', 'train_folder')
        self.patch_coor_file = cfg.get('TRAIN', 'patch_coor_file')
        self.patch_hdf5_folder = cfg.get('TRAIN', 'patch_hdf5_folder')
        ### file
        self.patch_hdf5_train_file_pre = cfg.get('TRAIN', 'patch_hdf5_train_file_pre')
        self.patch_hdf5_val_file_pre = cfg.get('TRAIN', 'patch_hdf5_val_file_pre')
        ### number
        self.patch_num_each_hdf5 = cfg.getint('TRAIN', 'patch_num_each_hdf5')
        self.model_info = cfg.getint('TRAIN', 'model_info')
        ### others
        self.model = cfg.get('TRAIN', 'model')
        self.model_pretrain = cfg.getboolean('TRAIN', 'model_pretrain')


        # self. = cfg.get('DEFAULT', '')
        # self. = cfg.get('DEFAULT', '')
        # self. = cfg.get('DEFAULT', '')
        # self. = cfg.get('DEFAULT', '')

    def check_dirs(self):
        self.check_dir(self.normal_data_folder)
        self.check_dir(self.tumor_data_folder)
        self.check_dir(self.tumor_anno_folder)
        self.check_dir(self.result_folder)
        self.check_dir(self.split_folder)
        self.check_dir(self.vis_ov_mask_folder)
        self.check_dir(self.vis_patch_folder)
        self.check_dir(self.vis_pos_patch_folder)
        self.check_dir(self.vis_neg_patch_folder)
        self.check_dir(self.patch_save_folder)
        # TRAIN
        self.check_dir(self.train_folder)
        self.check_dir(self.patch_coor_folder)
        self.check_dir(self.patch_hdf5_folder)
        # self.check_dir()
        # self.check_dir()

    def check_dir(self, dir):
        if not os.path.exists(dir):
            os.mkdir(dir)


if __name__ == '__main__':
    cfg = config()
    print(cfg)
