from torch.utils.data import Dataset
import patch_preprocess_fun
import slide_fun
import random
import config_fun
import hdf5_fun


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


class h5_dataloader(Dataset):
    def __init__(self, data_type = 'train', frac=1, file_name=None):
        cfg = config_fun.config()
        self._data_type = data_type
        self._compose = patch_preprocess_fun.get_train_val_compose()
        if file_name is None:
            self._data, self._label, self._name = hdf5_fun.get_all_data_label_name(cfg, data_type=data_type, frac=frac)
        else:
            self._data, self._label, self._name = hdf5_fun.h5_extract_data_label_name(cfg.patch_size, file_name)
        assert self._data.shape[0] == self._label.shape[0]

    def __getitem__(self, index):
        return self._compose(self._data[index]), self._label[index]

    def __len__(self):
        return self._label.shape[0]


class gm_cls_DataLoader(Dataset):
    def __init__(self, input_list, slide, patch_size):
        # super(testDataLoader, self).__init__()
        self._patch_size = patch_size
        self._data = input_list
        self._slide = slide
        self._compose = patch_preprocess_fun.get_gm_compose()

    def __getitem__(self, index):
        img = self._slide.read_region(self._data[index]['raw'], 0,
                                (self._patch_size, self._patch_size))
        return self._compose(img)

    def __len__(self):
        return len(self._data)