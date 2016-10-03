import subprocess
import argparse
import os.path

parser = argparse.ArgumentParser()
parser.add_argument('list')
parser.add_argument('command')
parser.add_argument('--data_root', default='/home/yanhang/Documents/research/DynamicStereo/data/working/')
args = parser.parse_args()

current_dir = os.path.dirname(os.path.realpath(__file__))

listfile = open(args.list)

datasets = listfile.readlines()

for data in datasets:
    name = data.split()[0]
    cmd = 'cd {}/{} && {}'.format(args.data_root, name, args.command)
    print cmd
    subprocess.call(cmd, shell=True)
    subprocess.call('cd ' + current_dir, shell=True)