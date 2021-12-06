from argparse import ArgumentParser
import time
from datetime import datetime
import os
import shutil
import logging
import random
import json
import mkl
import multiprocessing
import copy

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
from torch import nn
from torch import optim

import numpy as np
import matplotlib.pyplot as plt

from model import DataLoader, SharedConvNet, NoShareConvNet

def get_config():
    parser = ArgumentParser()
    parser.add_argument("--hidden_channels", type=int, default=101)
    parser.add_argument("--out_channels", type=int, default=225)
    parser.add_argument("--kernel_size1", type=int, default=3)
    parser.add_argument("--kernel_size2", type=int, default=3)
    parser.add_argument("--kernel_size3", type=int, default=3)
    parser.add_argument("--intermediate_dim", type=int, default=512)
    parser.add_argument("--batch_norm", type=int, default=1)
    parser.add_argument("--pooling", type=int, default=1)
    parser.add_argument("--resnet", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--convdropout", type=float, default=0.0)
    parser.add_argument("--model_name", type=str, default='SharedConvNet')
    
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--nb_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--lr_milestones", type=str, default="800,1200,1800,2500,4000,6000,8000")
    parser.add_argument("--save_dir", type=str, default='none')
    parser.add_argument("--seed", type=int, default=1) 
    parser.add_argument("--float64", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    if args.save_dir.lower() == "none":
        args.save_dir='./results/'+args.model_name+'/'+now.strftime("%m-%d-%Y-%H-%M-%S")+\
            f'{args.hidden_channels}_{args.out_channels}_{args.kernel_size1}_{args.kernel_size2}_{args.kernel_size3}_{args.intermediate_dim}_{args.batch_norm}_{args.pooling}_{args.resnet}_{args.dropout}_{args.convdropout}'
    else:
        args.save_dir = "./results/" + args.save_dir

    if args.float64:
        torch.set_default_dtype(torch.float64)
    else:
        torch.set_default_dtype(torch.float32)

    os.makedirs(args.save_dir, exist_ok=True)
    with open(args.save_dir + '/config.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    file_handler = logging.FileHandler(os.path.join(args.save_dir, "log.txt"), mode="w")
    if args.save_dir:
        for filename in os.listdir('./'):
            if '.sh' in filename or \
               '.swb' in filename or \
               '.py' in filename:
                   if filename == '.pylint.d':
                       continue
                   if '__pycache__' in filename:
                       continue
                   shutil.copy(filename, args.save_dir)
        shutil.copytree('./model', args.save_dir+'/model', dirs_exist_ok=True, ignore=shutil.ignore_patterns('*__pycache__*'))
        logger.addHandler(file_handler)

    return args

def set_seed(seed):
    """set random seed
    """
    logging.info(f'random {seed=}')
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':

    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    now = datetime.now()
    
    config = get_config()
    set_seed(config.seed)
    
    device = torch.device(config.device) if torch.cuda.is_available() else torch.device('cpu')
    
    loss_writer = open(config.save_dir+"/loss.txt", 'a+')
    accu_writer = open(config.save_dir+"/accuracy.txt", 'a+')
    
    dataloader = DataLoader(config.batch_size)
    trainloader, testloader, classes = dataloader()
    
    model = globals()[config.model_name](3, config.hidden_channels, config.out_channels,
                                         config.kernel_size1, config.kernel_size2, config.kernel_size3,
                                         config.intermediate_dim, 10, config.batch_norm, 
                                         config.pooling, config.resnet, config.dropout, config.convdropout)
    logging.info(f'number of parameters: {model.count_params()}')
    model = model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), config.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(map(int, config.lr_milestones.split(','))), gamma=config.gamma)
    
    for epoch in range(config.nb_epochs):
        model.train()
        running_loss = 0.
        for i, data in enumerate(trainloader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            if (i+1) % 25 == 0:
                logging.info(f'epoch={epoch+1}, i={i+1}, loss={running_loss/25}')
                for item in [epoch+1, i+1, running_loss]:
                    loss_writer.write("%s " % item)
                loss_writer.write('\n')
                running_loss = 0.0
                
        model.eval()
        
        train_correct = 0
        train_total = 0.
        
        with torch.no_grad():
            for i, data in enumerate(trainloader):
                if i == 10000:
                    break
                images, labels = data
                outputs = model(images.to(device))
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.shape[0]
                train_correct += (predicted == labels.to(device)).sum().item()
        train_accu = train_correct / train_total
        logging.info(f'epoch={epoch+1}, train accuracy={train_accu * 100}%')
        
        test_correct = 0
        test_total = 0.
        
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = model(images.to(device))
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.shape[0]
                test_correct += (predicted == labels.to(device)).sum().item()
        test_accu = test_correct / test_total
        logging.info(f'epoch={epoch+1}, test accuracy={test_accu * 100}%')
        
        for item in [epoch+1, train_accu, test_accu]:
            accu_writer.write("%s " % item)
        accu_writer.write('\n')
        
        loss_writer.flush()
        accu_writer.flush()
        
    torch.save(model.state_dict(), config.save_dir+"/model.pt")
        
        

