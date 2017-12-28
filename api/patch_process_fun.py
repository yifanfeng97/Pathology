from torchvision import transforms
import config_fun
import sys

def get_input_size(cfg):
    if cfg.model == 'vgg':
        return 224
    elif cfg.model == 'googlenet':
        return 299
    elif cfg.model == 'resnet':
        return 224
    else:
        print('not support the model: ' + cfg.model)
        sys.exit(-1)


def get_compose():
    cfg = config_fun.config()
    input_size = get_input_size(cfg)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    compose = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomSizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    return compose



