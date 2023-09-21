import util_v4 as util
from distoptim import *
import os
import numpy as np
import time
import argparse
import sys
import torch.nn as nn

parser = argparse.ArgumentParser(description='CIFAR-10 baseline')
parser.add_argument('--name','-n', 
                    default="default", 
                    type=str, 
                    help='experiment name, used for saving results')
parser.add_argument('--model', 
                    default="res", 
                    type=str, 
                    help='neural network model')
parser.add_argument('--alpha', 
                    default=0.2, 
                    type=float, 
                    help='control the non-iidness of dataset')
parser.add_argument('--gmf', 
                    default=0, 
                    type=float, 
                    help='global (server) momentum factor')
parser.add_argument('--lr', 
                    default=0.1, 
                    type=float, 
                    help='client learning rate')
parser.add_argument('--momentum', 
                    default=0.0, 
                    type=float, 
                    help='local (client) momentum factor')
parser.add_argument('--bs', 
                    default=512, 
                    type=int, 
                    help='batch size on each worker/client')
parser.add_argument('--rounds', 
                    default=200, 
                    type=int, 
                    help='total coommunication rounds')
parser.add_argument('--localE', 
                    default=98, 
                    type=int, 
                    help='number of local epochs')
parser.add_argument('--print_freq', 
                    default=100, 
                    type=int, 
                    help='print info frequency')
parser.add_argument('--size', 
                    default=8, 
                    type=int, 
                    help='number of local workers')
parser.add_argument('--rank', 
                    default=0, 
                    type=int, 
                    help='the rank of worker')
parser.add_argument('--seed', 
                    default=1, 
                    type=int, 
                    help='random seed')
parser.add_argument('--save', '-s', 
                    action='store_true', 
                    help='whether save the training results')
parser.add_argument('--p', '-p', 
                    action='store_true', 
                    help='whether the dataset is partitioned or not')
parser.add_argument('--NIID',
                    action='store_true',
                    help='whether the dataset is non-iid or not')
parser.add_argument('--pattern',
                    type=str, 
                    help='pattern of local steps')
parser.add_argument('--optimizer', 
                    default='local', 
                    type=str, 
                    help='optimizer name')
parser.add_argument('--mu', 
                    default=0, 
                    type=float, 
                    help='mu parameter in fedprox')
parser.add_argument('--savepath',
                    default='./results/',
                    type=str,
                    help='directory to save exp results')
parser.add_argument('--datapath',
                    default='./data/',
                    type=str,
                    help='directory to load data')


args = parser.parse_args()

# Load data
# DataRatios: relative sample size of clients
# rank: client index
# size: total number of clients

train_loader_list, test_loader_list, DataRatios_list, optimizer_list = ([] * args.size, )*4

model = util.select_model(10, args)
criterion = nn.CrossEntropyLoss()

algorithms = {
    'fedavg': FedProx, # with args.mu = 0
    'fedprox': FedProx, # with args.mu > 0
    'fednova': FedNova,
}

selected_algorithm = algorithms[args.optimizer]


def init_clients():
    for rank in range(args.size):
        train_loader_list[rank], test_loader_list[rank], DataRatios_list[rank] = util.partition_dataset(rank, args.size, args)
        optimizer_list[rank] = selected_algorithm(model.parameters(),
                                    lr = args.lr,
                                    mu = args.mu,
                                    gmf = args.gmf,
                                    ratio=DataRatios_list[rank],
                                    momentum=args.momentum,
                                    nesterov = False,
                                    weight_decay=1e-4)


def simulate():
    for rnd in range(args.rounds):

        # Performs local epochs on each client
        for rank in range(args.size):
            # Decide number of local updates per client
            local_epochs = update_local_epochs(args.pattern, rank, rnd)
            tau_i = local_epochs * len(train_loader_list[rank])
            logging.info("local epochs {} iterations {}"
                      .format(local_epochs, tau_i))           
                                
            # Perform local updates
            for t in local_epochs:
                train(model, criterion, optimizer_list[rank], train_loader_list[rank], t)
                    
            # Aggregate local changes  
            optimizer_list[rank].average()
            


def update_local_epochs(pattern, rank, rnd):
    if pattern == "constant":
        return args.localE

    if pattern == "uniform_random":
        np.random.seed(2020+rank+rnd+args.seed)
        return np.random.randint(low=2, high=args.localE, size=1)[0]    

def train(model, criterion, optimizer, loader, epoch):

    model.train()

    losses = util.Meter(ptag='Loss')
    top1 = util.Meter(ptag='Prec@1')

    for batch_idx, (data, target) in enumerate(loader):
        # data loading
        data = data.cuda(non_blocking = True)
        target = target.cuda(non_blocking = True)

        # forward pass
        output = model(data)
        loss = criterion(output, target)

        # backward pass
        loss.backward()

        # gradient step
        optimizer.step()
        optimizer.zero_grad()

        # write log files
        train_acc = util.comp_accuracy(output, target)
        

        losses.update(loss.item(), data.size(0))
        top1.update(train_acc[0].item(), data.size(0))

        if batch_idx % args.print_freq == 0 and args.save:
            logging.debug('epoch {} itr {}, '
                         'rank {}, loss value {:.4f}, train accuracy {:.3f}'
                         .format(epoch, batch_idx, rank, losses.avg, top1.avg))

            with open(args.out_fname, '+a') as f:
                print('{ep},{itr},'
                      '{loss.val:.4f},{loss.avg:.4f},'
                      '{top1.val:.3f},{top1.avg:.3f},-1'
                      .format(ep=epoch, itr=batch_idx,
                              loss=losses, top1=top1), file=f)

    with open(args.out_fname, '+a') as f:
        print('{ep},{itr},'
              '{loss.val:.4f},{loss.avg:.4f},'
              '{top1.val:.3f},{top1.avg:.3f},-1'
              .format(ep=epoch, itr=batch_idx,
                      loss=losses, top1=top1), file=f)


