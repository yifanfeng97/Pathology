import sys

def get_model(cfg):
    model = None
    if cfg.model == 'googlenet':
        from models import inception_v3
        model = inception_v3.inception_v3(pretrained = cfg.model_pretrain)
    elif cfg.model == 'vgg':
        from models import vgg
        if cfg.model_info == 19:
            model = vgg.vgg19_bn(pretrained = cfg.model_pretrain)
        elif cfg.model_info == 16:
            model = vgg.vgg16_bn(pretrained = cfg.model_pretrain)
    elif cfg.model == 'resnet':
        from models import resnet
        if cfg.model_info == 18:
            model = resnet.resnet18(pretrained= cfg.model_pretrain)
        elif cfg.model_info == 34:
            model = resnet.resnet34(pretrained= cfg.model_pretrain)
        elif cfg.model_info == 50:
            model = resnet.resnet50(pretrained= cfg.model_pretrain)
        elif cfg.model_info == 101:
            model = resnet.resnet101(pretrained= cfg.model_pretrain)
    if model is None:
        print('not support :' + cfg.model)
        sys.exit(-1)
    return model