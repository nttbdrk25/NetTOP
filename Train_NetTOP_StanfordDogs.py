#!/usr/bin/env python3

import argparse
import sys
import os
#import inspect
import time
import torch
import torch.nn
import torch.optim
import torch.optim.lr_scheduler
import torch.utils.data
import torchvision.transforms
import torchvision.datasets

#from pathlib import Path
#sys.path.append(str(Path('.').absolute().parent))
from models.datasets import *
from models.NetTOP import *
from models.cross_entropy import LabelSmoothingCrossEntropy
import writeLogAcc as wA

def get_args():
    """
    Parse the command line arguments.
    """
    parser = argparse.ArgumentParser(description='TickNet training script for CIFAR and fine-grained datasets.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-r', '--data-root', type=str, default='../../../datasets/StanfordDogs', help='Dataset root path.')
    #parser.add_argument('-d', '--dataset', choices=['cifar10', 'cifar100', 'dogs'], required=True, help='Dataset name.')
    parser.add_argument('-d', '--dataset', type=str, choices=['cifar10', 'cifar100', 'dogs'], default='dogs', help='Dataset name.')
    parser.add_argument('--download', action='store_true', help='Download the specified dataset before running the training.')    
    parser.add_argument('-g', '--gpu-id', default=1, type=int, help='ID of the GPU to use. Set to -1 to use CPU.')
    parser.add_argument('-j', '--workers', default=4, type=int, help='Number of data loading workers.')
    parser.add_argument('-b', '--batch-size', default=64, type=int, help='Batch size.')        
    parser.add_argument('-e', '--epochs', default=200, type=int, help='Number of total epochs to run.')
    parser.add_argument('-l', '--learning-rate', default=0.1, type=float, help='Initial learning rate.')
    parser.add_argument('-s', '--schedule', nargs='+', default=[100, 150, 180], type=int, help='Learning rate schedule (epochs after which the learning rate should be dropped).')    
    parser.add_argument('-m', '--momentum', default=0.9, type=float, help='SGD momentum.')
    parser.add_argument('-w', '--weight-decay', default=1e-4, type=float, help='SGD weight decay.')    
    parser.add_argument('--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    return parser.parse_args()


def get_device(args):
    """
    Determine the device to use for the given arguments.
    """
    if args.gpu_id >= 0:
        return torch.device('cuda:{}'.format(args.gpu_id))
    else:
        return torch.device('cpu')
    

def get_data_loader(args, train):
    """
    Return the data loader for the given arguments.
    """
    if args.dataset in ('cifar10', 'cifar100'):
        # select transforms based on train/val
        if train:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
            ])
        else:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
            ])
    
        # cifar10 vs. cifar100
        if args.dataset == 'cifar10':
            dataset_class = torchvision.datasets.CIFAR10
        else:
            dataset_class = torchvision.datasets.CIFAR100
            
    elif args.dataset in ('dogs',):
        # select transforms based on train/val
        if train:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(size=(256, 256)),
                torchvision.transforms.RandomCrop(224),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ColorJitter(0.4),
                torchvision.transforms.ToTensor()
            ])
        else:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(size=(256, 256)),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor()
            ])
    
        #dataset_class = models.datasets.StanfordDogs
        dataset_class = StanfordDogs
    
    else:
        raise NotImplementedError('Can\'t determine data loader for dataset \'{}\''.format(args.dataset))
    
    # trigger download only once
    if args.download:
        dataset_class(root=args.data_root, train=train, download=True, transform=transform)

    # instantiate dataset class and create data loader from it
    dataset = dataset_class(root=args.data_root, train=train, download=False, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True if train else False, num_workers=args.workers)    


def calculate_accuracy(output, target):
    """
    Top-1 classification accuracy.
    """
    with torch.no_grad():
        batch_size = output.shape[0]
        prediction = torch.argmax(output, dim=1)
        return torch.sum(prediction == target).item() / batch_size


def run_epoch(train, data_loader, model, criterion, optimizer, n_epoch, args, device):
    """
    Run one epoch. If `train` is `True` perform training, otherwise validate.
    """
    if train:
        model.train()
        torch.set_grad_enabled(True)
    else:
        model.eval()
        torch.set_grad_enabled(False)
    
    batch_count = len(data_loader)
    losses = []
    accs = []
    for (n_batch, (images, target)) in enumerate(data_loader):
        images = images.to(device)
        target = target.to(device)

        output = model(images)
        loss = criterion(output, target)        

        # record loss and measure accuracy
        loss_item = loss.item()
        losses.append(loss_item)
        acc = calculate_accuracy(output, target)
        accs.append(acc)

        # compute gradient and do SGD step
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        if (n_batch % 10) == 0:
            print('[{}]  epoch {}/{},  batch {}/{},  loss_{}={:.5f},  acc_{}={:.2f}%'.format('train' if train else ' val ', n_epoch + 1, args.epochs, n_batch + 1, batch_count, "train" if train else "val", loss_item, "train" if train else "val", 100.0 * acc))
    
    return (sum(losses) / len(losses), sum(accs) / len(accs))
            

def main():
    """
    Run the complete model training.
    """
    args = get_args()
    print('Command: {}'.format(' '.join(sys.argv)))
    args.gpu_id = 0
    device = get_device(args)
    print('Using device {}'.format(device))
    
    # print model with parameter and FLOPs counts    
    torch.autograd.set_detect_anomaly(True)             
    #arr_typesize_groups = [2,1,4]
    arr_typesize_groups = [1]
    
    for typesize in arr_typesize_groups:    
        strmode = 'NetTOP_groups_' + str(typesize)
        pathout = './checkpoints/StanfordDogs_NetTOP/' + strmode
        filenameLOG = pathout + '/' + strmode + '.txt'
        
        # get model        
        model = build_NetTOP(120, cifar=False,groups=typesize)
        model = model.to(device)
        
        print(model)
        
        print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
        
        # define loss function and optimizer
        train_loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1).to(device)        
        criterion = torch.nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.SGD(params=model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=args.schedule, gamma=0.1)
    
        # get train and val data loaders
        train_loader = get_data_loader(args=args, train=True)
        val_loader = get_data_loader(args=args, train=False)
        if args.evaluate:
            pathcheckpoint = "./checkpoints/StanfordDogs/checkpoint_120_6528.pth"
            if os.path.isfile(pathcheckpoint):
                print("=> loading checkpoint '{}'".format(pathcheckpoint))
                checkpoint = torch.load(pathcheckpoint)
                model.load_state_dict(checkpoint['model_state_dict'])                
                del checkpoint
            else:
                print("=> no checkpoint found at '{}'".format(pathcheckpoint))
                return
            m = time.time()
            (loss_val, acc_val) = run_epoch(train=False, data_loader=val_loader, model=model, criterion=criterion, optimizer=None, n_epoch=0, args=args, device=device)
            print('[{}], loss_{}={:.5f},  acc_{}={:.2f}%'.format(' validating: ', "val", loss_val, "val", 100.0 * acc_val))
            n = time.time()
            print((n-m)/3600)
            return
        # for each epoch...
        if not os.path.exists(pathout):
            os.makedirs(pathout)
        acc_val_max = None
        acc_val_argmax = None
        for n_epoch in range(args.epochs):
            current_learning_rate = optimizer.param_groups[0]['lr']
            print('Starting epoch {}/{},  learning_rate={}'.format(n_epoch + 1, args.epochs, current_learning_rate))
            
            # train            
            (loss_train, acc_train) = run_epoch(train=True, data_loader=train_loader, model=model, criterion=train_loss_fn, optimizer=optimizer, n_epoch=n_epoch, args=args, device=device)
    
            # validate
            (loss_val, acc_val) = run_epoch(train=False, data_loader=val_loader, model=model, criterion=criterion, optimizer=None, n_epoch=n_epoch, args=args, device=device)
            if (acc_val_max is None) or (acc_val > acc_val_max):
                acc_val_max = acc_val
                acc_val_argmax = n_epoch
                torch.save({"model_state_dict": model.state_dict()}, pathout + '/' + 'checkpoint_epoch{:>04d}_{:.2f}.pth'.format(n_epoch + 1,100.0 * acc_val_max))
    
            # adjust learning rate
            scheduler.step()
    
            # save the model weights
            #torch.save({"model_state_dict": model.state_dict()}, 'checkpoint_epoch{:>04d}.pth'.format(n_epoch + 1))
            
            # print epoch summary
            line = 'Epoch {}/{} summary:  loss_train={:.5f},  acc_train={:.2f}%,  loss_val={:.2f},  acc_val={:.2f}% (best: {:.2f}% @ epoch {})'.format(n_epoch + 1, args.epochs, loss_train, 100.0 * acc_train, loss_val, 100.0 * acc_val, 100.0 * acc_val_max, acc_val_argmax + 1)
            print('=' * len(line))
            print(line)
            print('=' * len(line))
            wA.writeLogAcc(filenameLOG,line)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Stopped')
        sys.exit(0)
    except Exception as e:
        print('Error: {}'.format(e))
        sys.exit(1)
