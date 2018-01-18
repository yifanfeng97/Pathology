from torch.utils.data import Dataset
import torch
import glob
import patch_preprocess_fun
import slide_fun
import random


class slides_dataloader(Dataset):
    def __init__(self, block, cfg):
        super(slides_dataloader, self).__init__()
        self.info = block['info']
        self.coors = block['coor']
        self.file_names = block['file_name']
        self.imgs = []
        self.patch_size = cfg.patch_size
        self.compose = patch_preprocess_fun.get_slide_compose()
        for file_name in self.file_names:
            self.imgs.append(slide_fun.AllSlide(file_name))
        random.shuffle(self.coors)

    def __getitem__(self, index):
        img_idx = self.coors[index][0]
        coor = self.coors[index][1]
        label = self.coors[index][2]
        patch = self.imgs[img_idx].read_region(coor, 0,
                        (self.patch_size, self.patch_size))
        return self.compose(patch), label

    def __len__(self):
        return len(self.coors)