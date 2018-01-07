import sys
import torch
import os
from api import hdf5_fun

def get_model(cfg, pretrained=True, load_param_from_folder=False):

    if load_param_from_folder:
        pretrained = False

    model = None
    num_classes = cfg.num_classes
    if cfg.model == 'googlenet':
        from models import inception_v3
        model = inception_v3.inception_v3(pretrained = pretrained, num_classes = num_classes)
    elif cfg.model == 'vgg':
        from models import vgg
        if cfg.model_info == 19:
            model = vgg.vgg19_bn(pretrained = pretrained, num_classes = num_classes)
        elif cfg.model_info == 16:
            model = vgg.vgg16_bn(pretrained = pretrained, num_classes = num_classes)
    elif cfg.model == 'resnet':
        from models import resnet
        if cfg.model_info == 18:
            model = resnet.resnet18(pretrained= pretrained, num_classes = num_classes)
        elif cfg.model_info == 34:
            model = resnet.resnet34(pretrained= pretrained, num_classes = num_classes)
        elif cfg.model_info == 50:
            model = resnet.resnet50(pretrained= pretrained, num_classes = num_classes)
        elif cfg.model_info == 101:
            model = resnet.resnet101(pretrained= pretrained, num_classes = num_classes)
    if model is None:
        print('not support :' + cfg.model)
        sys.exit(-1)

    if load_param_from_folder:
        print('loading pretrained model from {0}'.format(cfg.init_model_file))
        checkpoint = torch.load(cfg.init_model_file)
        model.load_state_dict(checkpoint['model_param'])

    return model


def save_checkpoint(model, output_path):
    ## if not os.path.exists(output_dir):
    ##    os.makedirs("model/")
    torch.save(model, output_path)

    print("Checkpoint saved to {}".format(output_path))

# do gradient clip
def clip_gradient(optimizer, grad_clip):
    assert grad_clip>0, 'gradient clip value must be greater than 1'
    for group in optimizer.param_groups:
        for param in group['params']:
            # gradient
            param.grad.data.clamp_(-grad_clip, grad_clip)


def save_model_and_optim(cfg, model, optimizer, epoch, best_prec1):
    path_checkpoint = os.path.join(cfg.checkpoint_folder, 'model_param.pth')
    # path_checkpoint = '{0}/{1}/model_param.pth'.format(cfg.checkpoint_folder, epoch)
    checkpoint = {}
    checkpoint['model_param'] = model.state_dict()

    save_checkpoint(checkpoint, path_checkpoint)

    # save optim state
    path_optim_state = os.path.join(cfg.checkpoint_folder, 'optim_state_best.pth')
    # path_optim_state = '{0}/{1}/optim_state_best.pth'.format(cfg.checkpoint_folder, epoch)
    optim_state = {}
    optim_state['epoch'] = epoch  # because epoch starts from 0
    optim_state['best_prec1'] = best_prec1
    optim_state['optim_state_best'] = optimizer.state_dict()
    save_checkpoint(optim_state, path_optim_state)
    # problem, should we store latest optim state or model, currently, we donot


def get_data(train, flag):
    data = hdf5_fun.h5_dataloader(train=train, flag=flag)
    dataLoader = torch.utils.data.DataLoader(data, batch_size=cfg.batch_size,
                                               shuffle=True, num_workers=int(cfg.workers))
    return dataLoader