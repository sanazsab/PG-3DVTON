import os
import os.path as osp
import torch
import torch.nn as nn
import torch.optim as optim
from config import Config
from torch.utils.data import DataLoader
from torchvision import transforms
from time import time
import datetime
import torch.backends.cudnn as cudnn
import numpy as np
from torchvision import utils
import sys
from data.regular_dataset import RegularDataset
from models.models import create_model
from models.generation_model import GenerationModel
from lib.geometric_matching_multi_gpu import GMM
#import cProfile
import re
import torchvision.models as models
#from torch.profiler import profile, record_function, ProfilerActivity

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVCIES'] = '0, 1, 2, 3, 4, 5, 6, 7, 8 , 9'

gpu_ids = len(os.environ['CUDA_VISIBLE_DEVCIES'].split(','))
cudnn.benchmark = True

augment = {}

if '0.4' in torch.__version__:
    augment['3'] = transforms.Compose([
                                # transforms.Resize(256),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5,), (0.5,0.5,0.5,))
        ]) # change to [C, H, W]
    augment['1'] = augment['3']

else:
    augment['3'] = transforms.Compose([
                            # transforms.Resize(256),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,0.5,0.5,), (0.5,0.5,0.5,))
    ]) # change to [C, H, W]

    augment['1'] = transforms.Compose([
                            # transforms.Resize(256),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,))
    ]) # change to [C, H, W]


def train(opt):
    model = GenerationModel(opt)
    dataset = RegularDataset(opt, augment=augment)
#    dataset = RegularDataset(opt)
    t0 = time()
    dataloader = DataLoader(
                        dataset,
                        shuffle=True,
                        drop_last=False,
                        num_workers=opt.num_workers,
                        batch_size=opt.batch_size_t,
                        pin_memory=True
    )
    t1 = time()
    print('After dataloader:', t1-t0)
    print('the length of dataset is %d'%len(dataset))
    for epoch in range(opt.start_epoch, opt.epoch):
        torch.cuda.empty_cache()
        print('current G learning_rate is : ', model.get_learning_rate(model.optimizer_G))
        print('train_mode: ', opt.train_mode)
        if opt.train_mode != 'gmm' and opt.train_mode != 'depth': #and opt.train_mode != 'gmm2':
            print('current D learning_rate is : ', model.get_learning_rate(model.optimizer_D))
#        elif opt.train_mode == 'depth' and opt.add_gan_loss:
#            print('current D learning_rate is : ', model.get_learning_rate(model.optimizer_FND))
#            print('current D learning_rate is : ', model.get_learning_rate(model.optimizer_BND))
        for i, data in enumerate(dataloader):
            t3= time()
            print('before loading:', t3-t0)
            model.set_input(opt, data)
            t4= time()
            print('loading:' , t4 - t3)
            model.optimize_parameters(opt)
            t5 = time()
            print('Optimizing:' , t5 - t4)
            if i % opt.print_freq == 0:
                model.print_current_errors(opt, epoch, i)
            if i % opt.val_freq == 0:
                model.save_result(opt, epoch, epoch * len(dataloader) + i)
#            print(np.array(model.optimizer_G).shape)
            model.update_learning_rate(opt, model.optimizer_G, epoch)
            if opt.train_mode != 'gmm' and opt.train_mode != 'depth': # and opt.train_mode != 'gmm2':
                model.update_learning_rate(opt, model.optimizer_D, epoch)

 #           if opt.train_mode == 'depth' and opt.add_gan_loss: 
 #               model.update_learning_rate(opt, model.optimizer_FND, epoch)
 #               model.update_learning_rate(opt, model.optimizer_BND, epoch)
        if epoch % opt.save_epoch_freq == 0:
            model.save_model(opt, epoch)
                
if __name__ == "__main__":
    opt = Config().parse()
    train(opt)

#    cProfile.run('train(opt)')
#    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
#        with record_function("model_inference"):
#            train(opt)
#    print(prof.key_averages().table(sort_by="cpu_time_total"))
