from __future__ import print_function, division
import visdom
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torch.utils.model_zoo as model_zoo
import time
from os.path import join, exists, isdir
from os import makedirs, listdir
import argparse
import copy
from PIL import Image
from matplotlib import pyplot as plt
from collections import namedtuple
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

def plot_kernels(tensor, num_cols=8):
    num_kernels = tensor.shape[0]
    num_rows = 1 + num_kernels // num_cols
    fig = plt.figure(figsize=(num_cols,num_rows))
    for i in range(tensor.shape[0]):
        ax1 = fig.add_subplot(num_rows,num_cols,i+1)
        ax1.imshow(tensor[i], cmap='gray')
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    return plt

######################################################################################################################################
######################################################################################################################################

parser = argparse.ArgumentParser(description='Visualize weights of VGG model using Visdom.')
parser.add_argument('model_name', metavar='model-name', help='Names of dataset on which model was trained. It will be used as suffix for model directory.', type=str)
parser.add_argument('model_dir', metavar='model-dir', help='Directory where model for dataset is to be found.', type=str)
parser.add_argument('--batchnorm', help='Whether Batch Normalization was used.', action='store_true')
parser.add_argument('--metrics', help='Whether Metrics should be displayed.', action='store_true')
parser.add_argument('--visualize', help='Whether to visualize', action='store_true')
parser.add_argument('-l', '--layer', help='Relu Layer number. Default 0. choices are 0-4.', nargs='?', type=int, const=0, default=0, choices=range(5))
parser.add_argument('-f', '--filter', help='Filter number. Default 0.', nargs='?', type=int, const=0, default=0)
args = parser.parse_args()

if not isdir(join(args.model_dir, args.model_name)):
    raise Exception('Given model directory does not exist')

model_dir = join(args.model_dir, args.model_name)

if args.metrics:
    vis = visdom.Visdom(server='http://10.4.16.22', env="metrics")
    metrics = {'tr': dict(), 'vl': dict()}
    names = set()
    print(listdir(model_dir))
    files = [x for x in listdir(model_dir) if x.startswith('vgg-tr-') or x.startswith('vgg-vl-')]
    print(files)
    for metric in files:
        name = metric.split('-')[-1].split('.')[0]
        phase = metric.split('-')[-2]
        names.add(name)
        file = join(model_dir, metric)
        y = [float(value if not value.startswith('tensor') else value.split('(')[1].split(',')[0]) for value in open(file)]
        x = [x for x in range(len(y))]
        metrics[phase][name] = {'x': x, 'y': y}
    for name in names:
        for phase in metrics.keys():
            if name in metrics[phase]:
                plt.plot(metrics[phase][name]['x'], metrics[phase][name]['y'], label = phase)
        plt.xlabel('Epochs')
        plt.ylabel(name.upper())
        plt.title('Model ' + name.upper())
        plt.legend()
        vis.matplot(plt)
        plt.close()

######################################################################################################################################
# GPU and Creating Model
if args.visualize:
    class Vgg16(torch.nn.Module):
        def __init__(self, model):
            super(Vgg16, self).__init__()
            features = list(model.features)[:30]
            self.features = nn.ModuleList(features).eval()

        def forward(self, x):
            results = []
            for ii, model in enumerate(self.features):
                x = model(x)
                if ii in {3, 8, 15, 22, 29}:
                    results.append(x)

            vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
            return vgg_outputs(*results)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_conv = vgg16_bn(pretrained=True, dropout=[0.3,0.3]) if args.batchnorm else vgg16(pretrained=True, dropout=[0.3,0.3])
    new_model_conv = (list(model_conv.classifier.children())[:-1])
    new_model_conv.append(nn.Linear(4096, 2))
    model_conv.classifier = nn.Sequential(*new_model_conv)
    model_conv = nn.DataParallel(model_conv)
    model_conv.load_state_dict(torch.load(join(model_dir, 'vgg_wei_err.pth'), device))
    model = Vgg16(model_conv.module.double())
    print(model)

    vis = visdom.Visdom(server='http://10.4.16.22')

    img = None
    relus = dict()
    for x in ['cancer', 'normal']:
    	img = Image.open('../input_images/' + ('lung' if args.model_name.startswith('lung') else 'prostate') + '_' + x + '_01.jpg').resize((224,224))
    	vis.image(np.rollaxis(np.array(img), 2, 0), opts={'title': x})
    	inp = np.ndarray((1, 3, 224, 224), dtype=np.double)
    	inp[0] = np.rollaxis(np.array(img), 2, 0)
    	relus[x] = model(torch.from_numpy(inp))

    plot = dict()
    # Properties window
    properties = [
        {'type': 'select', 'name': 'Block', 'value': 0, 'values': [0, 1, 2, 3, 4]},
        {'type': 'select', 'name': 'Filter', 'value': 0, 'values': [0, 1, 2, 3, 4]}
    ]

    properties_window = vis.properties(properties, opts={'title': 'Properties'})

    def properties_callback(event):
    	if event['event_type'] == 'PropertyUpdate':
    		prop_id = event['propertyId']
    		value = event['value']
    		properties[prop_id]['value'] = int(value)
    		vis.properties(properties, win=properties_window)
    		visualize(properties[0]['value'], properties[1]['value'])

    vis.register_event_handler(properties_callback, properties_window)

    def visualize(block_num, filter_num):
        layer = list([2, 7, 14, 21, 28])[block_num]
        print('found layer')
        trained_feature = np.uint8((list(model_conv.module.features.children())[layer]).weight.double().data.numpy()*255)
        print(trained_feature.shape)
        properties[1]['values'] = [x for x in range(trained_feature.shape[0])]
        vis.properties(properties, win=properties_window)
        plot['weights'] = vis.matplot(plot_kernels(trained_feature[filter_num] if layer else trained_feature, int(np.sqrt(trained_feature.shape[0]))), win = plot['weights'] if "weights" in plot else None, opts={'title': 'Filter Components'})
        print('plotted weights')
        for x in ['cancer', 'normal']:
            block = relus[x][block_num].detach().numpy()
            print(block.shape)
            imgs = block[0]
            plot[x] = vis.matplot(plot_kernels(imgs, int(np.sqrt(imgs.shape[0]))), win=plot[x] if x in plot else None, opts={'title': 'All Filter Results for ' + x})
            plot[x+"_filter"] = vis.image(np.array(Image.fromarray(np.uint8(block[0][filter_num])).resize((224,224))), win=plot[x+"_filter"] if x+"_filter" in plot else None, opts={'title': 'Specific Filter Result for ' + x})
            print('plotted for', x, 'image')

    visualize(properties[0]['value'], properties[1]['value'])

    try:
    	input = raw_input  # for Python 2 compatibility
    except NameError:
    	pass
    	input('Waiting for callbacks, press enter to quit.')
