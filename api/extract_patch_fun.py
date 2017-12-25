import slide_fun
import config_fun

class single_img_process():
    def __init__(self, data, type, auto_save_patch = True):
        self._cfg = config_fun.config()
        self._file_name = data['data'][0]
        self._mask_files = data['data'][1]
        self._auto_save_patch = auto_save_patch
        self._type = type

        self._img = slide_fun.AllSlide(self._file_name)
        self._raw_mask = None
        self._raw_size = self._img.level_dimensions[0]

    def _2048_level(self):
        level = self._img.level_count -1
        while self._img.level_dimensions[level][0] < 2048 and \
            self._img.level_dimensions[level][1] < 2048:
            level -= 1
        return level

    def _generate_mask(self):
        self._ov_mask = None
        self._ov_mask_size = self._2048_level()


    def _save_random_mask_and_patch(self):
        pass

    def _get_train_patch(self):
        pass

    def _save_patch(self):
        pass



def extract(data, type, auto_save_patch = True):
    img = single_img_process(data, type, auto_save_patch)
    img._generate_mask()
    img._get_train_patch()