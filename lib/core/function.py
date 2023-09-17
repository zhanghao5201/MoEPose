# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import time
import logging
import os

import numpy as np
import torch

from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images


logger = logging.getLogger(__name__)


def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        outputs = model(input)

        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        if isinstance(outputs, list):
            loss = criterion(outputs[0], target, target_weight)
            for output in outputs[1:]:
                loss += criterion(output, target, target_weight)
        else:
            output = outputs
            loss = criterion(output, target, target_weight)

        # loss = criterion(output, target, target_weight)

        # compute gradient and do update step
        optimizer.zero_grad()
        #print(loss.device)
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

            
            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)


def validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, mode='student'):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    print(mode)
    #mode='student'

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    k=0
    with torch.no_grad():
        end = time.time()
        for i, (input,input_new,target, target_weight, meta) in enumerate(val_loader):
            # compute output
            if (input_new==input).all() :
                k=k+1
            #print((input_new==input).all())
            input = input.cuda()
            input_new = input_new.cuda()
            if mode=='teacher':                
                _, _, _, outputs = model(input)
            elif mode=='student':
                #print("aaa")
                _, _, _, outputs = model(input_new)

            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            if config.TEST.FLIP_TEST:
                if mode=='teacher':
                    input_flipped = input.flip(3)
                    _, _, _,outputs_flipped = model(input_flipped)
                elif mode == 'student':
                    input_new_flipped = input_new.flip(3)
                    _, _, _, outputs_flipped = model(input_new_flipped)

                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()


                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                if mode=='teacher':
                    save_debug_images(config, input, meta, target, pred*4, output,
                                  prefix)
                elif mode=='student':
                    save_debug_images(config, input_new, meta, target, pred * 4, output,
                                      prefix)
        logger.info("k=({}".format(k))
        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )
        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

    return perf_indicator

def validateys(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, mode='student'):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    print(mode)
    #mode='student'

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    k=0
    with torch.no_grad():
        end = time.time()
        for i, (input,target, target_weight, meta) in enumerate(val_loader):
            # compute output
            
            #print((input_new==input).all())
            input = input.cuda()
            
            outputs = model(input)
            
            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            if config.TEST.FLIP_TEST:
                input_flipped = input.flip(3)
                outputs_flipped = model(input_flipped)
                
                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()


                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, target, pred*4, output,
                                  prefix)
        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )
        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

    return perf_indicator

def mutual_learning(config, train_loader, teacher,student, criterion, optimizer_t,optimizer_s, epoch,
          output_dir, tb_log_dir):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    criterion1=torch.nn.MSELoss(reduction='mean').cuda()
    #criterion1=criterion
    acc = AverageMeter()

    # switch to train mode
    teacher.train()
    student.train()

    end = time.time()
    for i, (input,input_new,target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        input = input.cuda()
        input_new = input_new.cuda()
        # compute output
        outputs_t = teacher(input)
        outputs_s = student(input_new)
        loss_dist_s=[]
        loss_dist_t=[]
        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        for j in range(3):
            for k in range(len(outputs_s[j])):
                if k == 0:
                    loss_dist_s.append(criterion1(outputs_s[j][k], outputs_t[j][k].data))
                    loss_dist_t.append(criterion1(outputs_t[j][k], outputs_s[j][k].data))
                else:
                    loss_dist_s[j] += criterion1(outputs_s[j][k], outputs_t[j][k].data)
                    loss_dist_t[j] += criterion1(outputs_t[j][k], outputs_s[j][k].data)
            loss_dist_s[j] = loss_dist_s[j] / (k + 1)
            loss_dist_t[j] = loss_dist_t[j] / (k + 1)

        loss_dist_last_s = criterion1(outputs_s[-1], outputs_t[-1].data)
        loss_dist_last_t = criterion1(outputs_t[-1], outputs_s[-1].data)
        loss_ori_s = criterion(outputs_s[-1], target,target_weight)
        loss_ori_t = criterion(outputs_t[-1], target,target_weight)

        #loss_s = 0.001*(loss_dist_s[1] + loss_dist_s[2] + loss_dist_last_s) + loss_ori_s
        #loss_t = 0.00001*( loss_dist_last_t) + loss_ori_t
        loss_s = 0.001*(loss_dist_s[0] + loss_dist_s[1] + loss_dist_s[2] + loss_dist_last_s) + loss_ori_s
        loss_t = 0.00001*(loss_dist_t[0] + loss_dist_t[1] + loss_dist_t[2] + loss_dist_last_t) + loss_ori_t
        #loss_s = loss_dist_last_s + loss_ori_s
        #loss_t = loss_dist_last_t + loss_ori_t

        # loss = criterion(output, target, target_weight)

        # compute gradient and do update step
        optimizer_s.zero_grad()
        optimizer_t.zero_grad()
        loss_s.backward()
        loss_t.backward()
        optimizer_s.step()
        optimizer_t.step()

        # measure accuracy and record loss
        losses.update(loss_s.item(), input_new.size(0))

        _, avg_acc, cnt, pred = accuracy(outputs_s[-1].detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)           

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input_new, meta, target, pred*4,outputs_s[-1],
                              prefix)

def distilling(config, train_loader, teacher,student, criterion,optimizer_s, epoch,
          output_dir, tb_log_dir):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    lossesys = AverageMeter()
    #criterion1=torch.nn.MSELoss(reduction='mean').cuda()
    criterion1=torch.nn.SmoothL1Loss(reduction='mean').cuda()

    #criterion1=criterion
    acc = AverageMeter()

    # switch to train mode
    teacher.eval()
    student.train()
    #device=torch.device("cuda:4" )
    end = time.time()
    for i, (input,input_new,target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        input=input.cuda()
        input_new=input_new.cuda()        
        data_time.update(time.time() - end)

        # compute output
        outputs_t = teacher(input)        
        outputs_s = student(input_new)
        #print(len(outputs_s),outputs_s[0].shape,outputs_s[1].shape,outputs_s[2].shape,outputs_s[3].shape)
        loss_dist_s=[]

        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        for j in range(3):
            for k in range(len(outputs_s[j])):
                if k == 0:
                    loss_dist_s.append(criterion1(outputs_s[j][k], outputs_t[j][k]))
                else:
                    loss_dist_s[j] += criterion1(outputs_s[j][k], outputs_t[j][k])
            loss_dist_s[j] = loss_dist_s[j] / (k + 1)

        loss_dist_last_s = criterion1(outputs_s[-1], outputs_t[-1])

        loss_ori_s = criterion(outputs_s[-1], target,target_weight)
        #loss_s = 0.001*( loss_dist_s[2] + loss_dist_last_s) + loss_ori_s
        loss_s = 0.00001*(loss_dist_s[0] + loss_dist_s[1] + loss_dist_s[2] + loss_dist_last_s) + loss_ori_s
        #loss_s = loss_ori_s  loss_dist_s[0] + 
        #print("loss_s",loss_s,loss_ori_s)
        if loss_s=="nan":
            print(meta)


        #loss_s = loss_dist_last_s + loss_ori_s
        #loss_t = loss_dist_last_t + loss_ori_t

        # loss = criterion(output, target, target_weight)

        # compute gradient and do update step
        optimizer_s.zero_grad()
        loss_s.backward()
        optimizer_s.step()

        # measure accuracy and record loss
        losses.update(loss_s.item(), input_new.size(0))
        lossesys.update(loss_ori_s.item(), input_new.size(0))

        _, avg_acc, cnt, pred = accuracy(outputs_s[-1].detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                    'Lossys {loss.val:.5f} ({Lossys.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, Lossys=lossesys,acc=acc)
            logger.info(msg)

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input_new, meta, target, pred*4,outputs_s[-1],
                              prefix)



# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
