from __future__ import print_function

import argparse
import socket
import time

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import numpy as np
import random
from models import model_pool
from models.util import create_model

from dataset.mini_imagenet import MetaImageNet
from dataset.tiered_imagenet import MetaTieredImageNet
from dataset.cifar import MetaCIFAR100
from eval.meta_eval import meta_test
from dataset.transform_cfg import transforms_test_options, transforms_list
from eval_fewshot import parse_option

def main():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    opt = parse_option()
    train_trans, test_trans = transforms_test_options[opt.transform]
    meta_testloader = DataLoader(MetaImageNet(args=opt, partition='test',
                                              train_transform=train_trans,
                                              test_transform=test_trans,
                                              fix_seed=False),
                                 batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                 num_workers=opt.num_workers)
    meta_valloader = DataLoader(MetaImageNet(args=opt, partition='val',
                                             train_transform=train_trans,
                                             test_transform=test_trans,
                                             fix_seed=False),
                                batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                num_workers=opt.num_workers)
    n_cls = 64
    model = create_model(opt.model, n_cls, opt.dataset)
    ckpt = torch.load(opt.model_path)
    model.load_state_dict(ckpt['model'])

    if torch.cuda.is_available():
        model = model.cuda()
        cudnn.benchmark = True

    start = time.time()
    test_acc, test_std = meta_test(model, meta_testloader, use_logit=False, classifier='original_avrithis')
    test_time = time.time() - start
    print('test_acc: {:.4f}, test_std: {:.4f}, time: {:.1f}'.format(test_acc, test_std, test_time))

if __name__ == '__main__':
    main()