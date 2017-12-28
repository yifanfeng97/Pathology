import sys
import torch

def get_model(cfg):
    model = None
    num_classes = cfg.num_classes
    if cfg.model == 'googlenet':
        from models import inception_v3
        model = inception_v3.inception_v3(pretrained = cfg.model_pretrain, num_classes = num_classes)
    elif cfg.model == 'vgg':
        from models import vgg
        if cfg.model_info == 19:
            model = vgg.vgg19_bn(pretrained = cfg.model_pretrain, num_classes = num_classes)
        elif cfg.model_info == 16:
            model = vgg.vgg16_bn(pretrained = cfg.model_pretrain, num_classes = num_classes)
    elif cfg.model == 'resnet':
        from models import resnet
        if cfg.model_info == 18:
            model = resnet.resnet18(pretrained= cfg.model_pretrain, num_classes = num_classes)
        elif cfg.model_info == 34:
            model = resnet.resnet34(pretrained= cfg.model_pretrain, num_classes = num_classes)
        elif cfg.model_info == 50:
            model = resnet.resnet50(pretrained= cfg.model_pretrain, num_classes = num_classes)
        elif cfg.model_info == 101:
            model = resnet.resnet101(pretrained= cfg.model_pretrain, num_classes = num_classes)
    if model is None:
        print('not support :' + cfg.model)
        sys.exit(-1)
    return model


def save_checkpoint(model, output_path):
    ## if not os.path.exists(output_dir):
    ##    os.makedirs("model/")
    torch.save(model, output_path)

    print("Checkpoint saved to {}".format(output_path))