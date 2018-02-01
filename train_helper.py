import sys
import torch
import os
from api import hdf5_fun
import random
import glob
import json
from api import patch_fun
import numpy as np
import os
from api import dataloader_fun


def get_model(cfg, pretrained=True, load_param_from_ours=False):

    if load_param_from_ours:
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

    if load_param_from_ours:
        print('loading pretrained model from {0}'.format(cfg.init_model_file))
        checkpoint = torch.load(cfg.init_model_file)
        model.load_state_dict(checkpoint['model_param'])

    model.cuda()
    print('shift model to parallel!')
    model = torch.nn.DataParallel(model, device_ids=cfg.gpu_id)
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


def save_model_and_optim(cfg, model, optimizer, epoch, best_prec1, confusion_mat):
    path_checkpoint = os.path.join(cfg.checkpoint_folder, 'model_param.pth')
    # path_checkpoint = '{0}/{1}/model_param.pth'.format(cfg.checkpoint_folder, epoch)
    checkpoint = {}
    if isinstance(model, torch.nn.DataParallel):
        model_save = model.module
    elif isinstance(model, torch.nn.Module):
        model_save = model
    else:
        print('model save type error')
        sys.exit()
    checkpoint['model_param'] = model_save.cpu().state_dict()
    model_save.cuda()

    save_checkpoint(checkpoint, path_checkpoint)

    # save optim state
    path_optim_state = os.path.join(cfg.checkpoint_folder, 'optim_state_best.pth')
    # path_optim_state = '{0}/{1}/optim_state_best.pth'.format(cfg.checkpoint_folder, epoch)
    optim_state = {}
    optim_state['epoch'] = epoch  # because epoch starts from 0
    optim_state['best_prec1'] = best_prec1
    optim_state['optim_state_best'] = optimizer.state_dict()
    optim_state['best_confusion_matrix'] = confusion_mat
    save_checkpoint(optim_state, path_optim_state)
    # problem, should we store latest optim state or model, currently, we donot


def get_dataloader(data_type, frac=1, file_name=None, cfg=None):
    data = dataloader_fun.h5_dataloader(data_type=data_type, frac=frac, file_name=file_name)
    dataLoader = torch.utils.data.DataLoader(data, batch_size=cfg.batch_size,
                                               shuffle=True, num_workers=int(cfg.workers))
    return dataLoader


def get_slide_dataloader(block, cfg):
    data = dataloader_fun.slides_dataloader(block, cfg)
    dataLoader = torch.utils.data.DataLoader(data, batch_size=cfg.batch_size,
                                             shuffle=True, num_workers=int(cfg.workers))
    return dataLoader


def get_slide_idx_coor_label(idx, coor, patch_type, cfg):
    coor_ans = []
    # if isinstance(coor[0], tuple):
    #     if patch_type == 'pos':
    #         label = 1
    #     else:
    #         label = 0
    #     coor_ans = [(idx, c, label) for c in coor]
    # elif isinstance(coor[0], list):
    if patch_type == 'pos':
        shift = cfg.num_neg_classes
    else:
        shift = 0
    for label, co in enumerate(coor):
        for c in co:
            coor_ans.append((idx, c, shift + label))
    return coor_ans


def get_block(slides, cfg):
    file_names = []
    coors = []
    info = []
    for idx, slide in enumerate(slides):
        file_names.append(slide['data'][0])
        info.append(slide['info'])
        coor_dir = os.path.join(cfg.patch_coor_folder,
                'coor_'+os.path.basename(slide['data'][0].split('.')[0]+'.npy'))
        coor = patch_fun.get_coor(coor_dir)['patch']

        if slide['info'].endswith('tumor'):
            coor = get_slide_idx_coor_label(idx, coor['pos'], 'pos', cfg)
        elif slide['info'].endswith('normal'):
            coor = get_slide_idx_coor_label(idx, coor['neg'], 'neg', cfg)
        else:
            print('get block label error')
            sys.exit(0)
        coors.extend(coor)
    block = {'file_name':file_names,
             'coor': coors,
             'info': info}
    return block


def get_blocks(data_type, cfg):
    blocks = []
    all_slide = json.load(open(cfg.split_file, 'r'))
    num_each_block = cfg.train_slide_num_each_block
    slides = [s for s in all_slide if s['info'].startswith(data_type)]
    random.shuffle(slides)

    for idx in range(0, len(slides), num_each_block):
        block = get_block(slides[idx:np.min((idx+num_each_block, len(slides)))], cfg)
        blocks.append(block)
    return blocks


def train_slide_wise(train, model, criterion, optimizer, epoch, cfg):
    blocks = get_blocks('train', cfg)
    random.shuffle(blocks)
    for idx, block in enumerate(blocks):
        print('[%d/%d] training data from file:' % (idx + 1, len(blocks)))
        for f in block['file_name']:
            print(f)
        dataloader = get_slide_dataloader(block, cfg)
        train(dataloader, model, criterion, optimizer, epoch, cfg)


def validate_slide_wise(validate, model, criterion, epoch, cfg):
    prec1_sum = 0
    blocks = get_blocks('val', cfg)
    random.shuffle(blocks)
    confusion_mat = None
    for idx, block in enumerate(blocks):
        print('[%d/%d] validation data from file:' % (idx + 1, len(blocks)))
        for f in block['file_name']:
            print(f)
        dataloader = get_slide_dataloader(block, cfg)
        prec1, confusion_mat = validate(dataloader, model, criterion, epoch, confusion_mat, cfg)
        prec1_sum = prec1_sum + prec1
    return prec1_sum/len(blocks), confusion_mat


def train_file_wise(train, model, criterion, optimizer, epoch, cfg):
    file_name_list = hdf5_fun.get_h5_file_list('train', cfg)
    for idx, file_name in enumerate(file_name_list):
        print('training data from file: %s [%d/%d]' % (file_name, idx+1, len(file_name_list)))
        dataloader = get_dataloader('train', file_name=file_name, cfg=cfg)
        train(dataloader, model, criterion, optimizer, epoch, cfg)


def validate_file_wise(validate, model, criterion, epoch, cfg):
    prec1_sum = 0
    file_name_list = hdf5_fun.get_h5_file_list('val', cfg)
    for idx, file_name in enumerate(file_name_list):
        print('validation data from file: %s [%d/%d]' % (file_name, idx+1, len(file_name_list)))
        dataloader = get_dataloader('val', file_name=file_name, cfg=cfg)
        prec1_sum += validate(dataloader, model, criterion, epoch, cfg)
    return prec1_sum/len(file_name_list)