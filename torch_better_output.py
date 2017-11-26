# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 21:15:39 2017

@author: Administrator
"""
#import torch
from torch.autograd import Variable
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
        self.avg = self.sum / self.count
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
def betteroutput(model,inputs,labels):
    input_var=Variable(inputs.cuda())
    output = model(input_var)
    prec1, prec3 = accuracy(output.data, labels, topk=(1, 3))
    top1.update(prec1[0], inputs.size(0))
    top3.update(prec3[0], inputs.size(0))


losses = AverageMeter()
top1 = AverageMeter()
top3 = AverageMeter()