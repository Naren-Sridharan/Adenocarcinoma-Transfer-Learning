"""
Created on Thu Oct 23 11:27:15 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import numpy as np

from misc_functions import (get_example_params,
                            convert_to_grayscale,
                            save_gradient_images)
from gradcam import GradCam
from guided_backprop import GuidedBackprop
import argparse

def guided_grad_cam(grad_cam_mask, guided_backprop_mask):
    """
        Guided grad cam is just pointwise multiplication of cam mask and
        guided backprop mask

    Args:
        grad_cam_mask (np_arr): Class activation map mask
        guided_backprop_mask (np_arr):Guided backprop mask
    """
    cam_gb = np.multiply(grad_cam_mask, guided_backprop_mask)
    return cam_gb


if __name__ == '__main__':
    # Get params
    parser = argparse.ArgumentParser(description='Visualize Guided GradCam using Visdom.')
    parser.add_argument('model_dir', metavar='model-dir', help='Directory where model for dataset is to be found.', type=str, const=None, default=None)
    parser.add_argument('--batchnorm', help='Whether Batch Normalization was used.', action='store_true')
    parser.add_argument('-s', '--server', help='Server address. Default http://10.4.16.22 (i.e. node13).', nargs='?', type=str, const="http://10.4.16.22", default="http://10.4.16.22")
    parser.add_argument('-t', '--target', help='Target number. Default 0.', nargs='?', type=int, const=0, default=0)
    parser.add_argument('-l', '--layer', help='Target Layer number. Default 11.', nargs='?', type=int, const=11, default=11)
    args = parser.parse_args()

    # Get params
    target_example = args.target
    (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
        get_example_params(target_example, args.server, args.model_dir, args.batchnorm)
    # Grad cam
    gcv2 = GradCam(pretrained_model, target_layer=args.layer)
    # Generate cam mask
    cam = gcv2.generate_cam(prep_img, target_class)
    print('Grad cam completed')

    # Guided backprop
    GBP = GuidedBackprop(pretrained_model)
    # Get gradients
    guided_grads = GBP.generate_gradients(prep_img, target_class)
    print('Guided backpropagation completed')

    # Guided Grad cam
    cam_gb = guided_grad_cam(cam, guided_grads)
    save_gradient_images(cam_gb, file_name_to_export + '_GGrad_Cam')
    grayscale_cam_gb = convert_to_grayscale(cam_gb)
    save_gradient_images(grayscale_cam_gb, file_name_to_export + '_GGrad_Cam_gray')
    print('Guided grad cam completed')
