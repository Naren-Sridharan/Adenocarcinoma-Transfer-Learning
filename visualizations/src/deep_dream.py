"""
Created on Mon Nov 21 21:57:29 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
from PIL import Image

import torch
from torch.optim import SGD
from torchvision import models

from misc_functions import get_example_params, preprocess_image, recreate_image, save_image
import argparse

class DeepDream():
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """
    def __init__(self, model, selected_layer, selected_filter, im_path):
        self.model = model
        self.model.eval()
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.conv_output = 0
        # Generate a random image
        self.created_image = Image.open(im_path).convert('RGB')
        # Hook the layers to get result of the convolution
        self.hook_layer()
        # Create the folder to export images if not exists
        if not os.path.exists('../generated'):
            os.makedirs('../generated')

    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = grad_out[0, self.selected_filter]

        # Hook the selected layer
        self.model[self.selected_layer].register_forward_hook(hook_function)

    def dream(self):
        # Process image and return variable
        self.processed_image = preprocess_image(self.created_image, True)
        # Define optimizer for the image
        # Earlier layers need higher learning rates to visualize whereas layer layers need less
        optimizer = SGD([self.processed_image], lr=12,  weight_decay=1e-4)
        for i in range(1, 251):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = self.processed_image
            for index, layer in enumerate(self.model):
                # Forward
                x = layer(x)
                # Only need to forward until we the selected layer is reached
                if index == self.selected_layer:
                    break
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(self.conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(self.processed_image)
            # Save image every 20 iteration
            if i % 10 == 0:
                print(self.created_image.shape)
                im_path = '../generated/ddream_l' + str(self.selected_layer) + \
                    '_f' + str(self.selected_filter) + '_iter' + str(i) + '.jpg'
                save_image(self.created_image, im_path)


if __name__ == '__main__':
    # THIS OPERATION IS MEMORY HUNGRY! #
    # Because of the selected image is very large
    # If it gives out of memory error or locks the computer
    # Try it with a smaller image
    parser = argparse.ArgumentParser(description='Visualize GradCam using Visdom.')
    parser.add_argument('model_dir', metavar='model-dir', help='Directory where model for dataset is to be found.', type=str, const=None, default=None)
    parser.add_argument('--batchnorm', help='Whether Batch Normalization was used.', action='store_true')
    parser.add_argument('-s', '--server', help='Server address. Default http://10.4.16.22 (i.e. node13).', nargs='?', type=str, const="http://10.4.16.22", default="http://10.4.16.22")
    parser.add_argument('-t', '--target', help='Target number. Default 0.', nargs='?', type=int, const=0, default=0)
    parser.add_argument('-l', '--layer', help='Target Layer number. Default 11.', nargs='?', type=int, const=11, default=11)
    parser.add_argument('-f', '--filter', help='Target Layer number. Default 5.', nargs='?', type=int, const=5, default=5)
    args = parser.parse_args()

    # Get params
    cnn_layer = args.layer
    filter_pos = args.filter
    target_example = args.target
    (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
        get_example_params(target_example, args.server, args.model_dir, args.batchnorm)
    dd = DeepDream(pretrained_model.features, cnn_layer, filter_pos, '../input_images/' + file_name_to_export + '.jpg')
    # This operation can also be done without Pytorch hooks
    # See layer visualisation for the implementation without hooks
    dd.dream()
