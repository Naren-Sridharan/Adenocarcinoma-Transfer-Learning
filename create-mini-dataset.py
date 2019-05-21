from os.path import join, exists, isdir
from os import listdir, makedirs
from shutil import copy
import argparse


parser = argparse.ArgumentParser(description='Create Mini Dataset from existing data set with same ratio between classes')
parser.add_argument('dataset', metavar='dataset', help='Name of existing dataset', type=str)
parser.add_argument('data_path', metavar='data-path', help='Path to existing dataset', type=str)
parser.add_argument('mini_data_path', metavar='mini-data-path', help='Path to create new dataset', type=str)
parser.add_argument('-m', '--mini_size', help='Default value is 10000', nargs='?', type=int, const=10000, default=10000)
parser.add_argument('--only_train', help='Create for training only', action='store_true')
args = parser.parse_args()

phases = ['train'] if args.only_train else ['train', 'valid', 'testi']

print('creating mini dataset for', args.dataset, 'from', join(args.data_path, args.dataset), 'in', join(args.mini_data_path, args.dataset + '-mini'))

for phase in phases:
	phase_dir = join(args.data_path, args.dataset, phase)
	phase_mini_dir = join(args.mini_data_path, args.dataset + '-mini', phase)
	if not exists(phase_mini_dir):
		makedirs(join(phase_mini_dir, 'cancer'))
		makedirs(join(phase_mini_dir, 'normal'))

	files = dict()
	number = dict()

	for classname in ['cancer', 'normal']:
		files[classname] = listdir(join(phase_dir, classname))
		number[classname] = len(files[classname])

	for classname in ['cancer', 'normal']:
		upto = int((number[classname] / sum(number.values())) * args.mini_size)
		for file in files[classname][:upto]:
			copy(join(phase_dir, classname, file), join(phase_mini_dir, classname))
	print('Done with', phase, 'phase')
