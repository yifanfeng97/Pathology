import slide_fun
import config_fun
import json

def generate_patch(auto_save_patch = True):
    cfg = config_fun.config()
    with open(cfg.split_file) as f:
        split_data = json.load(f)

    train_data = filter(lambda item: item['info'] == 'train_tumor' or
                                    item['info'] == 'train_normal', split_data)
    val_data   = filter(lambda item: item['info'] == 'val_tumor' or
                                    item['info'] == 'val_normal', split_data)
    test_data  = filter(lambda item: item['info'] == 'test_tumor' or
                                    item['info'] == 'test_normal', split_data)



