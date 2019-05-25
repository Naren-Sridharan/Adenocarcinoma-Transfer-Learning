from __future__ import print_function, division
import time
from os.path import join, exists, isdir
from os import makedirs
import argparse
import copy
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torch.utils.model_zoo as model_zoo
import pickle

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path
######################################################################################################################################
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

def f2_score(y_true, y_pred):
    return fbeta_score(y_true, y_pred, 2)


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
######################################################################################################################################

parser = argparse.ArgumentParser(description='Finds probability of each patch in the dataset and stores it in a pickle file with each entry having a slide_name under which all of the slide\'s patches are stored with x position, y position and probability')
parser.add_argument('model_name', metavar='model-name', help='Names of dataset on which model was trained. Will be used as suffix for data and log directories.', type=str)
parser.add_argument('test_name', metavar='test-name', help='Names of datasets, comma separated without blank spaces. Will be used as suffix for the log directory where the model should be.', type=str)
parser.add_argument('data_dir', metavar='data-dir', help='Path to data directory where dataset folders are present.', type=str)
parser.add_argument('log_dir', metavar='log-dir', help='Path to log directory where logs and models for datasets are to be found.', type=str)
parser.add_argument('-s', '--size', help='Image size to be transformed into. Default value is 224', nargs='?', type=int, const=224, default=224)
parser.add_argument('-p', '--phase', help='Phase - Split to test on.', nargs='?', type=str, const='testi', default='testi', choices=['testi', 'train', 'valid'])
parser.add_argument('--batchnorm', help='Batch Normalization', action='store_true')
args = parser.parse_args()


if not isdir(join(args.data_dir, args.test_name)):
    raise Exception('Given test directory does not exist')

if not isdir(join(args.log_dir, args.model_name)):
    raise Exception('Given model directory does not exist')

model_dir = join(args.log_dir, args.model_name)
log_file = join(args.log_dir, args.model_name + "_on_" + args.test_name + ".txt")
data_dir = join(args.data_dir, args.test_name, args.phase)
size = args.size

######################################################################################################################################
# GPU and Creating Model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_conv = vgg16_bn(pretrained=True, dropout=[0.3,0.3]) if args.batchnorm else vgg16(pretrained=True, dropout=[0.3,0.3])

new_model_conv = (list(model_conv.classifier.children())[:-1])
new_model_conv.append(nn.Linear(4096, 2))
model_conv.classifier = nn.Sequential(*new_model_conv)
model_conv = nn.DataParallel(model_conv)
model_conv.load_state_dict(torch.load(join(model_dir, 'vgg_wei_err.pth'), device))
# new_model_conv = list(model_conv.module.classifier.children())
# new_model_conv.append(nn.Sigmoid())
# model_conv.classifier = nn.Sequential(*new_model_conv)
model_conv = nn.DataParallel(model_conv)
model_conv = model_conv.module.to(device)


print(model_conv)

######################################################################################################################################

data_transform = transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.596, 0.436, 0.586], [0.2066, 0.240, 0.186])
        ])

# Datasets
image_dataset = ImageFolderWithPaths(join(data_dir), data_transform)


dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=128, shuffle=False)

dataset_size = len(image_dataset)

class_names = image_dataset.classes

patch_annotator = dict()

pickle_file = join(model_dir, 'probability.pickle')

if not exists(pickle_file):
    open(pickle_file, 'wb+').close()

with open(pickle_file, 'rb+') as prob_file:
    try:
        patch_annotator = pickle.load(prob_file)
    except:
        patch_annotator = dict()

######################################################################################################################################

# Training Function
def prob_model(model, patch_annotator):
    since = time.time()
    
    model.eval()    

    for inputs, labels, paths in dataloader:

        # get the inputs
        inputs = inputs.to(device)
        labels = labels.to(device)

        torch.set_grad_enabled(False)
        outputs = nn.Sigmoid()(model(inputs))
        for path, prob in zip(paths, outputs.data.cpu().numpy()[:, 0]):
            slide_name = path.split('/')[-1]
            x = int(slide_name.split('_')[1])
            y = int(slide_name.split('_')[2])
            slide_name = slide_name.split('_')[0]
            if slide_name not in patch_annotator:
                patch_annotator[slide_name] = list()
            patch_annotator[slide_name].append((x, y, prob))
    print(len(patch_annotator.keys()), ' slides\' patches have been annotated.')
    with open(pickle_file, 'rb+') as prob_file:
        try:
            patch_annotator = pickle.dump(patch_annotator, prob_file, pickle.HIGHEST_PROTOCOL)
        except:
            patch_annotator = dict()
    time_elapsed = time.time() - since
    print('probabilities found in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

######################################################################################################################################

prob_model(model_conv, patch_annotator)
