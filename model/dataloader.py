from collections import namedtuple

import torch
import torchvision
import torchvision.transforms as transforms

class DataLoader:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        
        transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
        self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                     download=True, transform=transform)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size,
                                                       shuffle=True, num_workers=2)
        
        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                    download=True, transform=transform)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size,
                                                      shuffle=False, num_workers=2)
        
        self.classes = ('plane', 'car', 'bird', 'cat',
                        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    def __call__(self):
        dataloader = namedtuple('dataloader', ['trainloader', 'testloader', 'classes'])
        return dataloader(self.trainloader, self.testloader, self.classes)