import sys
sys.path.append('../')
import ConfigParser

class config():
    def __init__(self, cfg_file = '../config/path.cfg'):

        cfg = ConfigParser.SafeConfigParser()
        cfg.read(cfg_file)
        ## read content
        # default
        self.normal_data_floder = cfg.get('DEFAULT', 'normal_data_floder')
        self.tumor_data_folder = cfg.get('DEFAULT', 'tumor_data_folder')
        self.tumor_anno_folder = cfg.get('DEFAULT', 'tumor_anno_folder')
        self.result_folder = cfg.get('DEFAULT', 'result_folder')
        self.img_ext = cfg.get('DEFAULT', 'img_ext')
        # preprocessS
        self.test_frac = cfg.getfloat('DEFAULT', 'test_frac')
        self.val_normal = cfg.getint('DEFAULT', 'val_normal')
        self.val_tumor = cfg.getint('DEFAULT', 'val_tumor')
        self.split_folder = cfg.get('DEFAULT', 'split_folder')
        self.split_file = cfg.get('DEFAULT', 'split_file')
        self.test_file = cfg.get('DEFAULT', 'test_file')
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


if __name__ == '__main__':
    cfg = config()
    print(cfg)
