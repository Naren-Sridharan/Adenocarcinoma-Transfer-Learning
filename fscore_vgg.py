from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
from os.path import join, exists, isdir
from os import makedirs
import argparse
import copy
import os
from slackclient import SlackClient
from sys import argv
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

######################################################################################################################################
#defining model
__all__ = [
    'vgg16_bn', 'vgg16'
]


model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True, dropout=[0.5,0.5]):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout[0]),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout[1]),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model

def vgg16_bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
    return model

######################################################################################################################################
#enable slack and provide app token
slack_enabled = True
try:
    slack_token = 'xoxp-334353176721-490886077219-615006917143-46395d5a71b71520b754b6b7cafee48b'
    sc = SlackClient(slack_token)
    print('Slack Progress Notifier Enabled.')
except:
    slack_enabled = False

def update(message):
    print(message)
    if slack_enabled:
        sc.api_call(
          "chat.postMessage",
          #give channel id for slack bot
          channel='UEES2296F',
          text=message,
        )
######################################################################################################################################

parser = argparse.ArgumentParser(description='Constructs a pretrained imagenet model and fine-tunes it with the dataset provided. It also stores logs and models of the training in given log_dir')
parser.add_argument('name', metavar='name', help='Name of dataset. Will be used as suffix for data and lof directories', type=str)
parser.add_argument('data_dir', metavar='data-dir', help='Path to data directory where images are present in folders train, valid and testi, under respective classes cancer and normal', type=str)
parser.add_argument('log_dir', metavar='log-dir', help='Path to log directory where logs and final models will be stored', type=str)
parser.add_argument('-s', '--size', help='Image size to be transformed into. Default value is 224', nargs='?', type=int, const=224, default=224)
parser.add_argument('-l', '--lr', help='Learning rate. Default value is 1e-4', nargs='?', type=float, const=1e-4, default=1e-4)
parser.add_argument('-b', '--batch', help='Batch Size. Default value is 128', nargs='?', type=int, const=128, default=128)
parser.add_argument('-w', '--weidecay', help='Weight Decay. Default value is 0', nargs='?', type=float, const=0, default=0)
parser.add_argument('-d', '--dropout', help='Dropout rates. Default value is "0.5,0.5"', nargs='?', type=str, const="0.5,0.5", default="0.5,0.5")
parser.add_argument('-bo', '--betaone', help='Beta 1. Default value is 0.9', nargs='?', type=float, const=0.9, default=0.9)
parser.add_argument('-bt', '--betatwo', help='Beta 2. Default value is 0.999', nargs='?', type=float, const=0.999, default=0.999)
parser.add_argument('-e', '--epochs', help='Epochs. Default value is 15', nargs='?', type=int, const=15, default=15)
parser.add_argument('-f', '--finetune', help='Fine-tune from. Default from ImageNet.', nargs='?', type=str, const=None, default=None)
parser.add_argument('-n', '--numfreeze', help='Freeze until nth layer. Default None.', nargs='?', type=int, const=None, default=None)
parser.add_argument('--batchnorm', help='Batch Normalization', action='store_true')
parser.add_argument('-r', '--ratio', help='Cancer:Normal ratio. Default 1.', nargs='?', type=float, const=1.0, default=1.0)
args = parser.parse_args()

#check existance of directories
if not isdir(join(args.data_dir, args.name)):
    raise Exception('Given data directory does not exist')

if not isdir(args.log_dir):
    raise Exception('Given log directory does not exist')
    
log_dir = join(args.log_dir, args.name + (('_with_' + args.finetune.split('/')[-2]) if args.finetune else ''))

if not isdir(log_dir):
    makedirs(log_dir)

#caluclate f2 score
def f2_score(y_true, y_pred):
    return fbeta_score(y_true, y_pred, 2)

#calculate f measure
def fbeta_score(y_true, y_pred, beta, eps=1e-9):
    beta2 = beta**2

    y_pred = y_pred.float()
    y_true = y_true.float()

    true_positive = (y_pred * y_true).sum()
    precision = true_positive.div(y_pred.sum().add(eps))
    recall = true_positive.div(y_true.sum().add(eps))

    return [torch.mean(
        (precision*recall).
        div(precision.mul(beta2) + recall + eps).
        mul(1 + beta2)), precision, recall]
        

######################################################################################################################################

# GPU and Creating Model
# creating model instance
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_conv = vgg16_bn(pretrained=True, dropout=[float(x) for x in args.dropout.split(',')]) if args.batchnorm else vgg16(pretrained=True, dropout=[float(x) for x in args.dropout.split(',')])

new_model_conv = (list(model_conv.classifier.children())[:-1])
new_model_conv.append(nn.Linear(4096, 2))
model_conv.classifier = nn.Sequential(*new_model_conv)

model_conv = nn.DataParallel(model_conv)

#finetune
finetune = ''
if args.finetune:
    finetune = ' with ' + args.finetune.split('/')[-2]
    update('Fine-tuning ' + args.name + finetune)  
    model_conv.load_state_dict(torch.load(join(args.finetune, 'vgg_wei_err.pth'), device))
model_conv = model_conv.to(device)

#freezing layers
if args.numfreeze:
    update('Freezing ' + str(args.numfreeze) + ' layers')
    for name, param in model_conv.named_parameters():
        print(name)
        if name.split('.')[1] == 'classifier' or int(name.split('.')[-2]) > args.numfreeze:
            break
        param.requires_grad = False

print(model_conv)
weight = args.ratio / (1 + args.ratio)
weight = torch.tensor([1 - weight, weight])
weight = weight.to(device)
criterion = nn.CrossEntropyLoss(weight)
vcriterion = nn.CrossEntropyLoss()
update('Training to start with args: ' + str(argv))

#Adam optimizer restricted to caculate gradients only for layers that are not frozen 
optimizer_conv = optim.Adam(filter(lambda p: p.requires_grad, model_conv.parameters()), lr=args.lr, weight_decay= args.weidecay, betas=(args.betaone, args.betatwo))

#Learning rate scheduler
exp_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_conv, 'min', patience=2, verbose=True, factor=0.1)

######################################################################################################################################
# data transforms and loading
data_dir = join(args.data_dir, args.name)
size = args.size

data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.596, 0.436, 0.586], [0.2066, 0.240, 0.186])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize([0.596, 0.436, 0.583], [0.2066, 0.240, 0.186]),    
        ]),
    }


# Datasets
image_datasets = {x: datasets.ImageFolder(join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'valid']}


dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch,
                                              shuffle=True)
               for x in ['train', 'valid']}


dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}

class_names = image_datasets['train'].classes

######################################################################################################################################

# Training Function
def train_model(model, t_criterion, v_criterion, optimizer, scheduler, num_epochs):
    curr_epoch = 0
    if exists(join(log_dir , "epoch_status.txt")):
        f = open(join(log_dir , "epoch_status.txt"), "r")
        curr_epoch = int(f.read()) + 1
        f.close()
        mdir = join(log_dir, 'vgg_wei_f2.pth')
        mdir = mdir if exists(mdir) else join(log_dir, 'vgg_wei_err.pth')
        model.load_state_dict(torch.load(mdir))
    update('Starting to train ' + args.name + finetune + ' model from ' + str(curr_epoch) + ' epoch.')
    epoch_loss = 0
    epoch_f2 = 0
    epoch_precision = 0
    epoch_recall = 0
    epoch_acc = 0
    since = time.time()
    best_model_wts = {x: model.state_dict() for x in ['f2', 'err']}
    best_f2 = 0.0
    best_error = 2.0

    for epoch in range(curr_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        info = 'Epoch ' + str(epoch) + ' for ' + args.name + finetune + ' model:\n'
        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            
            k = 0
            running_loss = 0.0
            running_f2 = 0.0
            running_precision = 0.0
            running_recall = 0.0
            running_corrects = 0

            # Train Phase

            if phase == 'train':
                model.train()
                print(phase)
                print(time.time()-since)
                for inputs, labels in dataloaders[phase]:
                    if k % 100 == 0:
                        print(k / 100)
                        f = open(join(log_dir, "status_18.txt"), "a+")
                        f.write(phase + "-" + str(k / 100))
                        f.write("\n")
                        f.close()
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)

                        _, preds = torch.max(outputs.data, 1)
                        loss = t_criterion(outputs, labels)
                        f2, precision, recall = f2_score(labels, preds)

                    # computing loss
                    loss.backward()
                    optimizer.step()

                    f = open(join(log_dir , "vgg-tr.txt"), "a+")
                    f.write(str(loss.item()))
                    f.write("\n")
                    f.close()
                    
                    k += 1

                    running_loss += loss.item()*inputs.size(0)
                    running_f2 += f2.item()*inputs.size(0)
                    running_precision += precision.item()*inputs.size(0)
                    running_recall += recall.item()*inputs.size(0)
                    running_corrects += torch.sum(preds.data == labels.data) 

                epoch_loss = (running_loss*1.0) / dataset_sizes[phase]
                epoch_f2 = (running_f2*1.0) / dataset_sizes[phase]
                epoch_precision = (running_precision*1.0) / dataset_sizes[phase]
                epoch_recall = (running_recall*1.0) / dataset_sizes[phase]
                epoch_acc = (running_corrects.double()) / dataset_sizes[phase]
                    
                f = open(join(log_dir , "vgg-tr-loss.txt"), "a+")
                f.write(str(epoch_loss))
                f.write("\n")
                f.close()
                f = open(join(log_dir , "vgg-tr-f2.txt"), "a+")
                f.write(str(epoch_f2))
                f.write("\n")
                f.close()
                f = open(join(log_dir , "vgg-tr-precison.txt"), "a+")
                f.write(str(epoch_precision))
                f.write("\n")
                f.close()
                f = open(join(log_dir , "vgg-tr-recall.txt"), "a+")
                f.write(str(epoch_recall))
                f.write("\n")
                f.close()
                f = open(join(log_dir , "vgg-tr-acc.txt"), "a+")
                f.write(str(epoch_acc.item()))
                f.write("\n")
                f.close()

            # Validation Phase

            if phase == 'valid':
                
                model.eval()
                print(phase)
                cn = 0
                run_cor = 0    

                for inputs, labels in dataloaders[phase]:

                    k += 1
                    if k % 100 == 0:
                        print(k / 100)
                    f = open(join(log_dir , "status_18.txt"), "a+")
                    f.write(phase+"-"+str(k/100))
                    f.write("\n")
                    f.close()

                # get the inputs
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)

                        _, preds = torch.max(outputs.data, 1)
                        loss = t_criterion(outputs, labels)
                        f2, precision, recall = f2_score(labels, preds)

                    f = open(join(log_dir , "vgg-vl.txt"), "a+")
                    f.write(str(loss.item()))
                    f.write("\n")
                    f.close()

                    running_loss += loss.item()*inputs.size(0)
                    running_f2 += f2.item()*inputs.size(0)
                    running_precision += precision.item()*inputs.size(0)
                    running_recall += recall.item()*inputs.size(0)
                    running_corrects += torch.sum(preds.data == labels.data)

                    if cn == 16:
                        f = open(join(log_dir , "validation-class"), "a+")
                        f.write(str(run_cor))
                        f.write("\n")
                        f.close()
                        run_cor = 0
                        cn = 0

                    if cn < 16:
                        run_cor += torch.sum(preds.data == labels.data)
                        cn = cn + 1

                epoch_loss = (running_loss*1.0) / dataset_sizes[phase]
                epoch_f2 = (running_f2*1.0) / dataset_sizes[phase]
                epoch_precision = (running_precision*1.0) / dataset_sizes[phase]
                epoch_recall = (running_recall*1.0) / dataset_sizes[phase]
                epoch_acc = (running_corrects.double()) / dataset_sizes[phase]

                scheduler.step(epoch_loss)
                f = open(join(log_dir , "vgg-vl-loss.txt"), "a+")
                f.write(str(epoch_loss))
                f.write("\n")
                f.close()
                f = open(join(log_dir , "vgg-vl-f2.txt"), "a+")
                f.write(str(epoch_f2))
                f.write("\n")
                f.close()
                f = open(join(log_dir , "vgg-tr-precison.txt"), "a+")
                f.write(str(epoch_precision))
                f.write("\n")
                f.close()
                f = open(join(log_dir , "vgg-tr-recall.txt"), "a+")
                f.write(str(epoch_recall))
                f.write("\n")
                f.close()
                f = open(join(log_dir , "vgg-vl-acc.txt"), "a+")
                f.write(str(epoch_acc.item()))
                f.write("\n")
                f.close()   

            print('{} Loss: {:.4f} F2: {:.4f} acc: {:.4f} precision: {:.4f} recall: {:.4f}'.format(
                phase, epoch_loss, epoch_f2, epoch_acc, epoch_precision, epoch_recall))

            # deep copy the model
            if phase == 'valid':
                if epoch_f2 > best_f2:
                    best_f2 = epoch_f2
                    best_model_wts['f2'] = copy.deepcopy(model.state_dict())

                elif epoch_loss < best_error:
                    best_error = epoch_loss
                    best_model_wts['err'] = copy.deepcopy(model.state_dict())
            print('Best val F2: {:4f}'.format(best_f2))
            print('Least val Err: {:4f}'.format(best_error))
            f = open(join(log_dir , "status_18.txt"), "a+")
            f.write("best f2 - " + str(best_f2))
            f.write("\n")
            f.write("least err - " + str(best_error))
            f.write("\n")
            f.close()
            info += phase + '\nF2: ' + str(epoch_f2) + '\nerr: ' + str(epoch_loss) + '\nacc: ' + str(epoch_acc.item()) + '\npre: ' + str(epoch_precision) + '\nre: ' + str(epoch_recall) + '\n'
        if epoch == 0:
            time_elapsed = time.time() - since
            update(phase + ' cycle takes {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        f = open(join(log_dir , "epoch_status.txt"), "w")
        f.write(str(epoch))
        f.close()
        for x in ['f2', 'err']:
            torch.save(best_model_wts[x], join(log_dir, 'vgg_wei_' + x + '.pth'))        
        update(info)
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val F2: {:4f}'.format(best_f2))

    # load best model weights
    model.load_state_dict(best_model_wts['f2'])
    update(args.name + ' model training completed.')
    return model

######################################################################################################################################
try:
    model_ft = train_model(model_conv, criterion, vcriterion, optimizer_conv, exp_lr_scheduler,
                           num_epochs=args.epochs)
except Exception as e:
    update(args.name + ' model training failed with exception: ' + str(e))
    raise e
