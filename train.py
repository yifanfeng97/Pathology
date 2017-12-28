# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import random
import time
import os
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel  # for multi-GPU training
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data

from torch.autograd import Variable
from api import hdf5_fun
from api import config_fun
import train_helper



cfg = config_fun.config()
model = train_helper.get_model(cfg)

parser = argparse.ArgumentParser()

# specify data and datapath
parser.add_argument('--dataset', default='modelnet40_v12', help='modelnet40_v12 | ?? ')
parser.add_argument('--data_dir', default='../data/data_h5', help='path to dataset')
# number of workers for loading data
parser.add_argument('--workers', type=int,
                    help='number of data loading workers, 0 means loading data using the main process(this)', default=2)
# loading data
parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
parser.add_argument('--pool_idx', type=int, default=21,
                    help='where to pool, avoid to pool at relu when it is inplace, mark dirty error')
# spcify optimization stuff
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')

parser.add_argument('--max_epochs', type=int, default=100, help='number of epochs to train for')
# parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--print_freq', type=int, default=25, help='number of iterations to print ')
parser.add_argument('--checkpoint_folder', default=None, help='check point path')
parser.add_argument('--model', type=str, default='', help='model path')

# cuda stuff
parser.add_argument('--gpu_id', type=str, default='1', help='which gpu to use, used only when ngpu is 1')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
# clamp parameters into a cube
parser.add_argument('--gradient_clip', type=float, default=0.01)

parser.add_argument('--have_aux', type=bool, default=False)
parser.add_argument('--input_views', type=int, default=12)

# resume training from a checkpoint
parser.add_argument('--init_model', default='', help="model to resume training")
parser.add_argument('--optim_state_from', default='', help="optim state to resume training")
parser.add_argument('--with_group', default=False, type=bool, help="whether with group")

opt = parser.parse_args()
print(opt)

if opt.checkpoint_folder is None:
    if with_group:
        if opt.have_aux:
            opt.checkpoint_folder = '%d_gnet_group_models_with_aux_checkpoint' % opt.input_views
        else:
            opt.checkpoint_folder = '%d_gnet_group_models_checkpoint' % opt.input_views
    else:
        if opt.have_aux:
            opt.checkpoint_folder = '%d_gnet_no_group_models_with_aux_checkpoint' % opt.input_views
        else:
            opt.checkpoint_folder = '%d_gnet_no_group_models_checkpoint' % opt.input_views

# make dir
os.system('mkdir {0}'.format(opt.checkpoint_folder))
print('mkdir %s' % opt.checkpoint_folder)

train_dataset = hdf5_fun.h5_dataloader(train= True)
val_dataset = hdf5_fun.h5_dataloader(train= False)


data_dir = r'../data'

print('number of train samples is: ', len(train_dataset))
print('number of test samples is: ', len(val_dataset))
print('finished loading data')

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

ngpu = int(opt.ngpu)
# opt.manualSeed = random.randint(1, 10000) # fix seed
opt.manualSeed = 123456

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
else:
    if ngpu == 1:
        print('so we use 1 gpu to training')
        print('setting gpu on gpuid {0}'.format(opt.gpu_id))

        if opt.cuda:
            torch.cuda.manual_seed(opt.manualSeed)

cudnn.benchmark = True
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)


def train(train_loader, model_gnet, criterion, optimizer, epoch, opt):
    """
    train for one epoch on the training set
    """
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    #############################################
    ## confusion table
    #############################################
    confusion = utils.ConfusionMatrix(40)

    # training mode
    model_gnet.train()

    end = time.time()
    for i, (inputs_12v, labels) in enumerate(train_loader):
        # bz x 12 x 3 x 224 x 224
        # re-view it to be : (bz * 12) x 3 x 224 x 224
        # note that inputs_12v.size(2) = 1, so we expand it to be 3
        inputs_12v = inputs_12v.view(inputs_12v.size(0) * inputs_12v.size(1), inputs_12v.size(2),
                                     inputs_12v.size(3), inputs_12v.size(4))
        labels = labels.long().view(-1)
        if isinstance(inputs_12v, torch.ByteTensor):
            inputs_12v = inputs_12v.float()
        # # expanding: (bz * 12) x 3 x 224 x 224
        #        inputs_12v = inputs_12v.expand(inputs_12v.size(0), 3,
        #            inputs_12v.size(2), inputs_12v.size(3))

        # byte tensor to float tensor
        # normalize data here instead of using clouse in dataset class, because it is
        # not format 12 x 1 x H x W in stead of C x H x W
        #        mean =  223.03979492188
        #        std = 1.0
        #        inputs_12v = utils.preprocess(inputs_12v, mean, std, False) # False means not do data augmentation

        inputs_12v = Variable(inputs_12v)
        labels = Variable(labels)

        # print(points.size())
        # print(labels.size())
        # shift data to GPU
        if opt.cuda:
            inputs_12v = inputs_12v.cuda()
            #            labels = labels.long().cuda() # must be long cuda tensor
            labels = labels.cuda()  # must be long cuda tensor

        # forward, backward optimize
        # (bz*12) X C x H x W

        preds = model_gnet(inputs_12v)  # bz x C x H x W
        #        print('labels:\n', labels.data)
        #        print('preds:\n', preds.data)
        # in pytorch, unlike torch, the label is 0-indexed (start from 0)
        #        labels = labels.sub_(1)

        # debug_here()
        # if labels.data.max() >= 40:
        #    debug_here()
        #   print('error')

        # if labels.data.min() < 0:
        #    debug_here()
        #    print('error')
        ###########################################
        ## add center loss
        ###########################################
        # 40 classes
        #        alpha = 0.1
        #        # preds as features
        #        center_loss, model_after_pool._buffers['centers'] = utils.get_center_loss(model_after_pool._buffers['centers'],
        #            preds, labels, alpha, 40)
        #
        #        # contrastive center loss
        #        print('size')
        #        print(preds.size(), labels.size())
        if opt.have_aux:
            preds, aux = preds
            loss_main = criterion(preds, labels)
            loss_aux = criterion(aux, labels)
            softmax_loss = loss_main + 0.3 * loss_aux
        else:
            softmax_loss = criterion(preds, labels)
        # center_loss_weight = 0 # set it to 0.1, can achive 91.625, set it to 0,
        #        loss = center_loss_weight * center_loss + softmax_loss
        loss = softmax_loss

        ###########################################
        ## measure accuracy
        ###########################################
        prec1 = utils.accuracy(preds.data, labels.data, topk=(1,))[0]
        losses.update(loss.data[0], preds.size(0))  # batchsize
        top1.update(prec1[0], preds.size(0))

        ###############################################
        ## confusion table
        ###############################################
        #        print('preds:')
        #        print(preds.data)
        #        print('labels:')
        #        print(labels.data)
        confusion.batchAdd(preds.data, labels.data)

        ###########################################
        ## backward
        ###########################################
        optimizer.zero_grad()
        loss.backward()
        #        utils.clip_gradient(optimizer, opt.gradient_clip)
        optimizer.step()

        # debug_here()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                loss=losses, top1=top1))
            # ###########################################
            # ## Log
            # ###########################################
            # # loss accuracy
            # step = epoch * len(train_loader) + i
            # loss_name= None
            # if opt.have_aux:
            #     loss_name = 'mixed_loss'
            # else:
            #     loss_name = 'loss'
            # info = { loss_name : loss.data[0],
            #         'accuracy': top1.avg}
            #
            # for tag, value in info.items():
            #     logger.scalar_summary(log_pre_name+'train/' + tag, value, step)
            # # parameters gradients
            # for tag, value in model_gnet.named_parameters():
            #     if not hasattr(value.grad, 'data'): continue
            #     tag = tag.replace('.', '/')
            #     logger.histo_summary(log_pre_name+'train/' + tag, to_np(value), step)
            #     logger.histo_summary(log_pre_name+'train/' + tag + '/grad', to_np(value.grad), step)
            # # images
    #####################################
    ## confusion table
    #####################################
    # debug_here()
    confusion.updateValids()
    print('mean class accuracy at epoch {0}: {1} '.format(epoch, confusion.mean_class_acc))


def validate(test_loader, model_gnet, criterion, optimizer, epoch, opt):
    """
    test for one epoch on the testing set
    """
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    ###############################
    ## confusion table
    ###############################
    confusion = utils.ConfusionMatrix(40)

    # training mode
    model_gnet.eval()

    end = time.time()
    for i, (inputs_12v, labels) in enumerate(test_loader):
        # bz x 12 x 1 x 224 x 224
        # re-view it to be : (bz * 12) x 3 x 224 x 224
        # note that inputs_12v.size(2) = 1, so we expand it to be 3
        inputs_12v = inputs_12v.view(inputs_12v.size(0) * inputs_12v.size(1), inputs_12v.size(2),
                                     inputs_12v.size(3), inputs_12v.size(4))
        labels = labels.long().view(-1)
        if isinstance(inputs_12v, torch.ByteTensor):
            inputs_12v = inputs_12v.float()
        # # expanding: (bz * 12) x 3 x 224 x 224
        #        inputs_12v = inputs_12v.expand(inputs_12v.size(0), 3,
        #            inputs_12v.size(2), inputs_12v.size(3))

        # byte tensor to float tensor
        # normalize data here instead of using clouse in dataset class, because it is
        # not format 12 x 1 x H x W in stead of C x H x W
        #        mean =  223.03979492188
        #        std = 1.0
        #        inputs_12v = utils.preprocess(inputs_12v, mean, std, False) # False means not do data augmentation

        inputs_12v = Variable(inputs_12v, volatile=True)
        labels = Variable(labels, volatile=True)

        # print(points.size())
        # print(labels.size())
        # shift data to GPU
        if opt.cuda:
            inputs_12v = inputs_12v.cuda()
            labels = labels.cuda()  # must be long cuda tensor

        # forward, backward optimize
        # (bz*12) X C x H x W
        preds = model_gnet(inputs_12v)  # bz x C x H x W

        # print(labels)
        # in pytorch, unlike torch, the label is 0-indexed (start from 0)
        #        labels = labels.sub_(1)

        # currently we do not use center loss here
        loss = criterion(preds, labels)

        ###########################################
        ## measure accuracy
        ###########################################
        prec1 = utils.accuracy(preds.data, labels.data, topk=(1,))[0]
        losses.update(loss.data[0], preds.size(0))  # batchsize
        top1.update(prec1[0], preds.size(0))

        ###############################################
        ## confusion table
        ###############################################
        confusion.batchAdd(preds.data, labels.data)

        # debug_here()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % opt.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(test_loader), batch_time=batch_time, loss=losses,
                top1=top1))
            # ###########################################
            # ## Log
            # ###########################################
            # # loss accuracy
            # step = epoch * len(test_loader) + i
            # info = { 'loss' : loss.data[0],
            #         'accuracy': top1.avg}
            #
            # for tag, value in info.items():
            #     logger.scalar_summary(log_pre_name+'test/'+tag, value, step)
            # # parameters gradients
            # for tag, value in model_gnet.named_parameters():
            #     tag = tag.replace('.', '/')
            #     logger.histo_summary(log_pre_name+'test/'+tag, to_np(value), step)
            # # images

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    #####################################
    ## confusion table
    #####################################
    confusion.updateValids()
    print('mean class accuracy at epoch {0}: {1} '.format(epoch, confusion.mean_class_acc))

    # print(tested_samples)
    return top1.avg


def main():
    global opt
    best_prec1 = 0
    # only used when we resume training from some checkpoint model
    resume_epoch = 0
    # train data loader
    # for loader, droplast by default is set to false
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchSize,
                                               shuffle=True, num_workers=int(opt.workers))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,
                                              shuffle=True, num_workers=int(opt.workers))

    # create model
    model_gnet = gnet.inception_v3(pretrained=True, aux_logits=opt.have_aux,
                                   transform_input=True, num_classes=40, \
                                   n_views=opt.input_views, with_group=with_group)
    if opt.init_model != '':
        print('loading pretrained model from {0}'.format(opt.init_model))
        checkpoint = torch.load(opt.init_model)
        model_gnet.load_state_dict(checkpoint['google_net'])

    print('google net model: ')
    print(model_gnet)

    # optimizer
    optimizer = optim.SGD(model_gnet.parameters(), opt.lr,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    # if we load model from pretrained, we need the optim state here
    if opt.optim_state_from != '':
        print('loading optim model from {0}'.format(opt.optim_state_from))
        optim_state = torch.load(opt.optim_state_from)

        resume_epoch = optim_state['epoch']
        best_prec1 = optim_state['best_prec1']
        optimizer.load_state_dict(optim_state['optim_state_best'])



        # cross entropy already contains logsoftmax
    criterion = nn.CrossEntropyLoss()

    if opt.cuda:
        print('shift model and criterion to GPU .. ')
        model_gnet = model_gnet.cuda()
        # define loss function (criterion) and pptimizer
        criterion = criterion.cuda()
    # optimizer
    # fix model_prev_pool parameters
    for p in model_gnet.parameters():
        p.requires_grad = False  # to avoid computation
    for p in model_gnet.fc.parameters():
        p.requires_grad = True  # to  computation
    if with_group:
        for p in model_gnet.fc_q.parameters():
            p.requires_grad = True
    if opt.have_aux:
        for p in model_gnet.AuxLogits.parameters():
            p.requires_grad = True  # to  computation

    for epoch in range(resume_epoch, opt.max_epochs):
        #################################
        # train for one epoch
        # debug_here()
        #################################
        # debug_here()

        if epoch >= 20:
            for p in model_gnet.parameters():
                p.requires_grad = True

        train(train_loader, model_gnet, criterion, optimizer, epoch, opt)

        #################################
        # validate
        #################################
        prec1 = validate(test_loader, model_gnet, criterion, optimizer, epoch, opt)

        ##################################
        # save checkpoints
        ##################################
        if best_prec1 < prec1:
            best_prec1 = prec1
            path_checkpoint = '{0}/model_best.pth'.format(opt.checkpoint_folder)
            checkpoint = {}
            checkpoint['google_net'] = model_gnet.state_dict()

            utils.save_checkpoint(checkpoint, path_checkpoint)

            # save optim state
            path_optim_state = '{0}/optim_state_best.pth'.format(opt.checkpoint_folder)
            optim_state = {}
            optim_state['epoch'] = epoch + 1  # because epoch starts from 0
            optim_state['best_prec1'] = best_prec1
            optim_state['optim_state_best'] = optimizer.state_dict()
            utils.save_checkpoint(optim_state, path_optim_state)
            # problem, should we store latest optim state or model, currently, we donot

        print('best accuracy: ', best_prec1)


if __name__ == '__main__':
    main()

