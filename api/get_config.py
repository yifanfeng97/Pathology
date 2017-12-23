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
        ## preprocess
        ### folder
        self.split_folder = cfg.get('PREPROCESS', 'split_folder')
        ### number
        self.test_frac = cfg.getfloat('PREPROCESS', 'test_frac')
        self.val_normal = cfg.getint('PREPROCESS', 'val_normal')
        self.val_tumor = cfg.getint('PREPROCESS', 'val_tumor')
        ### files
        self.split_file = cfg.get('PREPROCESS', 'split_file')
        self.test_file = cfg.get('PREPROCESS', 'test_file')
        ### others
        self.redividing = cfg.getboolean('PREPROCESS', 'redividing')
        # self. = cfg.get('DEFAULT', '')
        # self. = cfg.get('DEFAULT', '')
        # self. = cfg.get('DEFAULT', '')
        # self. = cfg.get('DEFAULT', '')
        # self. = cfg.get('DEFAULT', '')
        # self. = cfg.get('DEFAULT', '')
        # self. = cfg.get('DEFAULT', '')
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
        # self.check_dir()
        # self.check_dir()

    def check_dir(self, dir):
        if not os.path.exists(dir):
            os.mkdir(dir)


if __name__ == '__main__':
    cfg = config()
    print(cfg)
