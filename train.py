# -*- coding: utf-8 -*-

from __future__ import print_function
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
from api import meter


# os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu_id

def train(train_loader, model, criterion, optimizer, epoch, cfg):
    """
    train for one epoch on the training set
    """
    batch_time = meter.timemeter.TimeMeter()
    losses = meter.averagevaluemeter.AverageValueMeter()
    # top1 = meter.averagevaluemeter.AverageValueMeter()
    ap = meter.apmeter.APMeter()
    confusion = meter.confusionmeter.ConfusionMeter(cfg.num_classes)
    prec1 = meter.classerrormeter.ClassErrorMeter(accuracy=True)

    # training mode
    model.train()

    for i, (inputs_img, gt_labels) in enumerate(train_loader):
        batch_time.reset()
        prec1.reset()
        if isinstance(inputs_img, torch.ByteTensor):
            inputs_img = inputs_img.float()
        gt_labels = gt_labels.long().view(-1)
        inputs_img = Variable(inputs_img)
        gt_labels = Variable(gt_labels)

        # shift data to GPU
        inputs_img = inputs_img.cuda()
        gt_labels = gt_labels.cuda()  # must be long cuda tensor

        # forward, backward optimize
        preds = model(inputs_img)  # bz x C x H x W
        if cfg.model == 'googlenet':
            preds, aux = preds
            loss_main = criterion(preds, gt_labels)
            loss_aux = criterion(aux, gt_labels)
            softmax_loss = loss_main + 0.3 * loss_aux
        else:
            softmax_loss = criterion(preds, gt_labels)
        loss = softmax_loss

        ###########################################
        ## measure accuracy
        ###########################################
        prec1.add(preds.data, gt_labels.data)
        losses.add(loss.data[0], preds.size(0))  # batchsize
        # top1.add(prec1[0], preds.size(0))

        ###############################################
        ## confusion table
        ###############################################
        confusion.add(preds.data, gt_labels.data)

        ###########################################
        ## backward
        ###########################################
        optimizer.zero_grad()
        loss.backward()
        # train_helper.clip_gradient(optimizer, cfg.gradient_clip)
        optimizer.step()

        # debug_here()
        if i % cfg.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {3} \t'
                  'Loss {4}\t'.format(
                epoch, i, len(train_loader), batch_time.value(),
                losses.value()))
        # if i % cfg.print_freq == 0:
        #     print('Epoch: [{0}][{1}/{2}]\t'
        #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
        #         epoch, i, len(train_loader), batch_time=batch_time.value(),
        #         loss=losses.value(), top1=1))
            # ###########################################
            # ## Log
            # ###########################################
            # # loss accuracy
            # step = epoch * len(train_loader) + i
            # loss_name= None
            # if cfg.have_aux:
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
    print('mean class accuracy at epoch {0}: {1} '.format(epoch, confusion.value()))


def validate(val_loader, model, criterion, epoch, cfg):
    """
    test for one epoch on the testing set
    """
    batch_time = meter.timemeter.TimeMeter()
    losses = meter.averagevaluemeter.AverageValueMeter()
    # top1 = meter.averagevaluemeter.AverageValueMeter()
    ap = meter.apmeter.APMeter()
    confusion = meter.confusionmeter.ConfusionMeter(cfg.num_classes)
    prec1 = meter.classerrormeter.ClassErrorMeter(accuracy=True)


    # training mode
    model.eval()

    for i, (input_img, gt_labels) in enumerate(val_loader):
        batch_time.reset()
        prec1.reset()

        if isinstance(input_img, torch.ByteTensor):
            input_img = input_img.float()
        gt_labels = gt_labels.long().view(-1)
        input_img = Variable(input_img, volatile=True)
        gt_labels = Variable(gt_labels, volatile=True)

        # shift data to GPU
        input_img = input_img.cuda()
        gt_labels = gt_labels.cuda()  # must be long cuda tensor

        # forward, backward optimize
        preds = model(input_img)  # bz x C x H x W

        # print(labels)
        # in pytorch, unlike torch, the label is 0-indexed (start from 0)
        #        labels = labels.sub_(1)

        # currently we do not use center loss here
        loss = criterion(preds, gt_labels)

        ###########################################
        ## measure accuracy
        ###########################################
        prec1.add(preds.data, gt_labels.data)
        losses.add(loss.data[0], preds.size(0))  # batchsize
        # top1.add(prec1.value(), preds.size(0))

        ###############################################
        ## confusion table
        ###############################################
        confusion.add(preds.data, gt_labels.data)

        # measure elapsed time

        if i % cfg.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time.value(), loss=losses.value(),
                top1=1))
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

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=1))

    print('mean class accuracy at epoch {0}: {1} '.format(epoch, confusion.value()))

    # print(tested_samples)
    # return top1.value()
    return 1


def main():
    cfg = config_fun.config()
    model = train_helper.get_model(cfg)

    train_dataset = hdf5_fun.h5_dataloader(train=True)
    val_dataset = hdf5_fun.h5_dataloader(train=False)

    print('number of train samples is: ', len(train_dataset))
    print('number of test samples is: ', len(val_dataset))
    print('finished loading data')
    best_prec1 = 0
    # only used when we resume training from some checkpoint model
    resume_epoch = 0
    # train data loader
    # for loader, droplast by default is set to false
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size,
                                               shuffle=True, num_workers=int(cfg.workers))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.batch_size,
                                              shuffle=True, num_workers=int(cfg.workers))

    if cfg.resume_training:
        print('loading pretrained model from {0}'.format(cfg.init_model_file))
        checkpoint = torch.load(cfg.init_model_file)
        model.load_state_dict(checkpoint['model_param'])

    print('model: ')
    print(model)

    # optimizer
    optimizer = optim.SGD(model.parameters(), cfg.lr,
                          momentum=cfg.momentum,
                          weight_decay=cfg.weight_decay)

    # if we load model from pretrained, we need the optim state here
    if cfg.resume_training:
        print('loading optim model from {0}'.format(cfg.optim_state_file))
        optim_state = torch.load(cfg.optim_state_file)

        resume_epoch = optim_state['epoch']
        best_prec1 = optim_state['best_prec1']
        optimizer.load_state_dict(optim_state['optim_state_best'])

    criterion = nn.CrossEntropyLoss()

    print('shift model and criterion to GPU .. ')
    model = model.cuda()
    # define loss function (criterion) and pptimizer
    criterion = criterion.cuda()

    for epoch in range(resume_epoch, cfg.max_epoch):

        train(train_loader, model, criterion, optimizer, epoch, cfg)

        #################################
        # validate
        #################################
        prec1 = validate(val_loader, model, criterion, epoch, cfg)

        ##################################
        # save checkpoints
        ##################################
        if best_prec1 < prec1:
            best_prec1 = prec1
            path_checkpoint = '{0}/{1}/model_param.pth'.format(cfg.checkpoint_folder, epoch)
            checkpoint = {}
            checkpoint['model_param'] = model.state_dict()

            train_helper.save_checkpoint(checkpoint, path_checkpoint)

            # save optim state
            path_optim_state = '{0}/{1}/optim_state_best.pth'.format(cfg.checkpoint_folder, epoch)
            optim_state = {}
            optim_state['epoch'] = epoch  # because epoch starts from 0
            optim_state['best_prec1'] = best_prec1
            optim_state['optim_state_best'] = optimizer.state_dict()
            train_helper.save_checkpoint(optim_state, path_optim_state)
            # problem, should we store latest optim state or model, currently, we donot

        print('best accuracy: ', best_prec1)


if __name__ == '__main__':
    main()