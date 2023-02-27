#! /usr/bin/python
# -*- encoding: utf-8 -*-
#optimizer: 이전에 정의한 optimizer 변수명을 넣어준다.
#T_0: 첫번째 restart를 위해 몇번 iteration이 걸리는가?
#T_mult: restart 후에 T_i를 증가시키는 factor => T_0 = 8, T_mult = 2라면, T_1 = 16식, 다만 이제 문제는 
#eta_min: 최소 lr

import torch

def Scheduler(optimizer, test_interval, max_epoch, lr_decay, **kwargs):

	#sche_fn = torch.optim.lr_scheduler.StepLR(optimizer, step_size=test_interval, gamma=lr_decay)#
    sche_fn = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=test_interval, T_mult=2, eta_min=5e-6)
    lr_step = 'epoch'
    print('Initialised Cosine annealing LR scheduler')
    
    return sche_fn, lr_step