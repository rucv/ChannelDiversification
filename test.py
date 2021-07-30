import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from ptflops import get_model_complexity_info
from network import EmbeddingNet
#from network import ResNet

parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
parser.add_argument('--dataset', default='cifar100', type=str,
                    help='dataset (cifar10 [default] or cifar100)')

parser.add_argument('-b', '--batch-size', default=128, type=int,
                    help='mini-batch size (default: 128)')
print("PyTorch Version: ", torch.__version__)
#print("Torchvision Version: ", torchvision.__version__)
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='to use basicblock for CIFAR datasets (default: bottleneck)')

class ImageFolderWithPaths(datasets.CIFAR100):
     def __getitem__(self, index):
         # this is what ImageFolder normally returns 
         original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
         # the image file path
         #d_list = self.test_list
         #print(d_list[index])
         path = self.data[index]
         #print(str(pathi))
          
         # make a new tuple that includes original and the pat
         tuple_with_path = (original_tuple + (path,))
         return tuple_with_path

global args
args = parser.parse_args()

normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])


transform_train = transforms.Compose([
                transforms.ToTensor(),
        	transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
        						(4,4,4,4),mode='reflect').squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                ])

transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])

kwargs = {'num_workers': 1, 'pin_memory': True}
assert(args.dataset == 'cifar10' or args.dataset == 'cifar100')

train_loader = torch.utils.data.DataLoader(
        datasets.__dict__[args.dataset.upper()]('../data', train=True, download=True,
                         transform=transform_train),
        batch_size=args.batch_size, shuffle=True, **kwargs)

val_loader = torch.utils.data.DataLoader(
        ImageFolderWithPaths('../data', train=False, transform=transform_test),
        batch_size=args.batch_size, shuffle=False, **kwargs)


def test_model(model,val_loader):
    
    best_model_wts = torch.load("./runs/ResNet-28-10/model_best.pth.tar")     
    #checkpoint = torch.load('model_best.pth.tar')
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(best_model_wts['state_dict'])
    model.eval()
    #model = torch.nn.DataParallel(model).cuda()
    for param_tensor in model.state_dict():
       print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    total = 0
    correct = 0

    with torch.no_grad():
        for inputs, labels, path in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model.forward(inputs)
            _,preds = torch.max(outputs,1)
            c = (preds == labels).squeeze()
         
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            path_list = list(path)
            l = 0
           
    
    print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))      
  
  

model_ft = ResNet(args.bottleneck)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_ft = model_ft.to(device)

test_model(model_ft, val_loader)



   
        
        
        
   
    


  
      

