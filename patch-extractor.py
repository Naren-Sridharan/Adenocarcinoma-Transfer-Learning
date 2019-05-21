from PIL import Image, ImageChops, ImageStat
import openslide
from os.path import join, exists, isdir
from os import listdir, makedirs
import pickle
from shutil import move
from time import sleep
import argparse
import concurrent.futures
import multiprocessing
import random
import numpy as np

#count cores for parallel processing
core_count = multiprocessing.cpu_count()

#options
parser = argparse.ArgumentParser(description='Extracts patches of from the highest magnification(level 0) of the svs images in the  data_directory/slides/ containing .svs files only')
parser.add_argument('data_path', metavar='data-path', help='Path to data directory where slide images are present', type=str)
parser.add_argument('-p', '--patch-size', help='Default value is 512. Other choices are 256 and 1024. ', nargs='?', type=int, const=512, default=512, choices=[256, 512, 1024])
parser.add_argument('--equal', help='Equalize data sets', action='store_true')
args = parser.parse_args()

# basic settings
main_dir = args.data_path

#creating folders
if not isdir(join(main_dir,'slides')):
    raise Exception('Given path to folder does not exist')

slides_dir = join(main_dir, 'slides')

if not all([x.endswith('.svs') for x in listdir(slides_dir)]):
    raise Exception('Slides folder contains non-svs files, please remove them and try again')

patches_dir = join(main_dir, 'patches')

train_dir = join(patches_dir, 'train')
val_dir = join(patches_dir, 'valid')
test_dir = join(patches_dir, 'testi')

for data_dir in [train_dir, val_dir, test_dir]:
    if not exists(join(data_dir, 'cancer')):
        makedirs(join(data_dir, 'cancer'))
    if not exists(join(data_dir, 'normal')):
        makedirs(join(data_dir, 'normal'))

#initialize parameters
patch_size = args.patch_size
split = False
total_files = len(listdir(slides_dir))

patches_slides = []
min_patches = None


#find minumum number of patches to eqalize cancer and normal
if args.equal:
    print('Finding Minimun Number of Patches for Equalizing')
    for file in listdir(slides_dir):
        
        patch = None
        if int(file.split('-')[3][:2]) <= 10:
            #look only at normal patches
            continue
        try:
            osr = openslide.OpenSlide(join(slides_dir, file))
            if osr.get_thumbnail((512, 512)).getbbox() is None:
                #don't look at empty slides or slides without thumbnails as they are corrupted
                print(file)
                continue
        except:
            #exclue unreadable files
            print(file)
            continue
        #find dimesions of 40x magnification
        width, height = osr.dimensions
        #find number of patches that are obtainable from the image
        patches_slides.append(width*height//(patch_size**2))
    #find minimum patches required from each slide such that we get maximum number of equal patches overall
    total = len(patches_slides)
    patches_slides = sorted(patches_slides)
    total_patches = [n * (total - i) for i, n in enumerate(patches_slides)]
    n_variety = max(total_patches)
    n_variety -= n_variety%10000 + 10000
    min_patches = patches_slides[total_patches.index(min(total_patches, key=lambda x:abs(x-n_variety)))]
    min_patches -= min_patches%100
    print('Minimun Number of Patches: ' + str(min_patches))

#seggregate files into cancer and normal
cancer_slides = []
normal_slides = []
file_count = 0

for file in listdir(slides_dir):
    
    print('segregating file', file_count, end='\r')
    sleep(0.03)
    patch = None
    file_count += 1
    # open svs slide image
    try:
        osr = openslide.OpenSlide(join(slides_dir, file))
        if osr.get_thumbnail((512, 512)).getbbox() is None:
            print(file)
            continue
        width, height = osr.dimensions
        # exclude slides that dont have enough number of patches
        if args.equal and (width*height//(patch_size**2)) <= min_patches:
            print(file)
            continue
    except:
        print(file)
        continue

    if int(file.split('-')[3][:2]) <= 10:
        cancer_slides.append([file, 'cancer', len(cancer_slides) + 1])
    else:
        normal_slides.append([file, 'normal', len(normal_slides) + 1])
        
print(len(cancer_slides), 'cancerous,', len(normal_slides), 'normal')

#equalize slide count on both sides
if args.equal:
    print('Equalizing')
    if len(cancer_slides) > len(normal_slides):
        cancer_slides = random.sample(cancer_slides, k=len(normal_slides))
        cnt = 1
        for i in range(len(cancer_slides)):
            cancer_slides[i][2] = cnt
            cnt += 1
    else:
        normal_slides = random.sample(normal_slides, k=len(cancer_slides))
        cnt = 1
        for i in range(len(normal_slides)):
            normal_slides[i][2] = cnt
            cnt += 1
    print('Minimun Number of Patches: ' + str(min_patches))
    print(len(cancer_slides), 'cancerous,', len(normal_slides), 'normal')

slides = cancer_slides + normal_slides
total_count = {'cancer' : len(cancer_slides), 'normal': len(normal_slides)}

for slide in slides:
    file, result, number = slide
    total = total_count[result]
    slide.append(total)
    slide.append(train_dir if number <= int(total*0.70)             
                 else (val_dir if number <= int(total*0.85)
                       else test_dir)
                )

def extract_patch(slide):
    # extract patches    
    # whiteness limit
    whiteness_limit = (patch_size ** 2) / 2
    
    file, result, number, total, data_dir = slide 

    # open svs slide image
    try:
        osr = openslide.OpenSlide(join(slides_dir, file))
    except:
        print(file)
        return 0

    count = 0

    x_patches = osr.dimensions[0]-osr.dimensions[0]%patch_size
    y_patches = osr.dimensions[1]-osr.dimensions[1]%patch_size
    
    # slide across slide taking patches
    for x in range(0, x_patches, patch_size):
        for y in range(0, y_patches, patch_size):

            patch = osr.read_region(location=(x, y), level=0, size=(patch_size, patch_size)).convert('RGB')

            # alternative
            # get patch image stats
            white = all([w >= whiteness_limit 
                         for w in 
                         ImageStat.Stat(Image.eval(patch, lambda x: 1 if x >= 210 else 0)).sum]
                       )

            if white:
                continue

            # save into respective folder
            patch.save(join(data_dir, result, file[:file.find('.svs')] + '_{:06d}'.format(x) + '_{:06d}'.format(y) + '_{:06d}'.format(count) + '.jpg'))

            count += 1
            if args.equal and count >= min_patches:
                return count
    return count

with concurrent.futures.ProcessPoolExecutor(max_workers=core_count*2) as executor:
    for slide, patch_count in zip(slides, executor.map(extract_patch, slides)):
        continue
    
print("\n patch extraction completed.")
