#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch

def Scheduler(optimizer, test_interval, max_epoch, lr_decay, **kwargs):
    #sche_fn = torch.optim.lr_scheduler.StepLR(optimizer, step_size=test_interval, gamma=lr_decay)
	sche_fn = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr = 5e-6, max_lr = 1e-3, step_size_up=4, mode='triangular')
    
	lr_step = 'epoch'

	print('Initialised cyclic LR scheduler')

	return sche_fn, lr_step

