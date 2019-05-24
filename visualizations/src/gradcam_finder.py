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
from PIL import Image
from joblib import Parallel, delayed
import multiprocessing
from misc_functions import get_example_params, preprocess_image, apply_colormap_on_image
import argparse
from gradcam import GradCam

core_count = multiprocessing.cpu_count()
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

parser = argparse.ArgumentParser(description='Tests existing model against a test dataset. It also stores logs of the test in the given log_dir.')
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

pickle_file = join(model_dir, 'gradcam.pickle')

if not exists(pickle_file):
    open(pickle_file, 'wb+').close()

with open(pickle_file, 'rb+') as grad_file:
    try:
        patch_annotator = pickle.load(grad_file)
    except:
        patch_annotator = dict()

######################################################################################################################################
# Get params
target_example = 0
(original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
    get_example_params(target_example, "http://10.4.16.22", model_dir, args.batchnorm)

model_conv = nn.DataParallel(pretrained_model)
# Grad cam
grad_cam = GradCam(pretrained_model, target_layer=0)
def gen_heatmap(path, label):
    org_img = Image.open(path).convert('RGB').resize((224, 224))
    prep_img = preprocess_image(org_img)
    cam = grad_cam.generate_cam(prep_img, label)
    heatmap, heatmap_on_image = apply_colormap_on_image(org_img, cam, 'hsv')
    slide_name = path.split('/')[-1]
    x = int(slide_name.split('_')[1])
    y = int(slide_name.split('_')[2])
    slide_name = slide_name.split('_')[0]
    return [slide_name, (x, y, heatmap)]
# Training Function
def prob_model(model, patch_annotator):
    since = time.time()
    
    model.eval()    
    count = 0
    for inputs, labels, paths in dataloader:
        outputs = Parallel(n_jobs=core_count*2)(delayed(gen_heatmap)(path, label) for path, label in zip(paths, labels))
        print(outputs)
        exit()
        count += 128
        print(count, '/', dataset_size, 'patches annotated so far', end='\r')
    print(len(patch_annotator.keys()), ' slides\' patches have been annotated.')
    with open(pickle_file, 'rb+') as grad_file:
        try:
            patch_annotator = pickle.dump(patch_annotator, grad_file, pickle.HIGHEST_PROTOCOL)
        except:
            print('failed to dump')
    time_elapsed = time.time() - since
    print('gradcams found in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

######################################################################################################################################
print(model_conv)
prob_model(model_conv, patch_annotator)
