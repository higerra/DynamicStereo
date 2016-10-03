import subprocess
import argparse
import os.path

parser = argparse.ArgumentParser()
parser.add_argument('--source_ip', default='yanhang@172.16.20.23')
parser.add_argument('listfile', default='/home/yanhang/Documents/research/DynamicStereo/data/working/list_datasets.txt')

args = parser.parse_args()

data_root = '/home/yanhang/Documents/research/DynamicStereo/data/working/'

list_file = open(args.listfile)
datasets = list_file.readlines()

for data in datasets:
    info = data.split()
    name = info[0]
    command = 'scp -r {}:{}/{}/temp {}/{}/'.format(args.source_ip, data_root, name, data_root, name)
    print command
    subprocess.call(command, shell=True)

    command = 'scp -r {}:{}/{}/midres/prewarp {}/{}/midres/'.format(args.source_ip, data_root, name, data_root, name)
    print command
    subprocess.call(command, shell=True)

    command = 'scp {}:{}/{}/midres/depth*.depth {}/{}/midres/'.format(args.source_ip, data_root, name, data_root, name)
    print command
    subprocess.call(command, shell=True)