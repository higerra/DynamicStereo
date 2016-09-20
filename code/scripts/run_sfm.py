import subprocess
import argparse

data_dir = '/home/yanhang/Documents/research/DynamicStereo/data/working/'


datasets = ['data_arki', 'data_casino3', 'data_casino5', 'data_casino6', 'data_casino7', 'data_casino8', 'data_newyork', 'data_newyork2', 'data_newyork3', 'data_osaka2', 'data_shopping', 'data_statue1', 'data_vegas5_2', 'data_vegas6', 'data_vegas7', 'data_vegas15', 'data_vegas17', 'data_vegas18', 'data_vegas20', 'data_vegas22', 'data_vegas24', 'data_vegas29', 'data_vegas33']

sfm_exec = '/home/yanhang/Documents/research/DynamicStereo/code/build/SfM/SfM'

parser = argparse.ArgumentParser()
parser.add_argument('--reset', action='store_true', help='remove all computed result')
args = parser.parse_args()

for data in datasets:
    fullpath = data_dir + data;
    if args.reset:
        command = 'rm -rf {0}/images/* {0}/midres/* {0}/sfm/* {0}/temp/*'.format(fullpath)
        print command
        subprocess.call(command, shell=True)
    command = '{} {} --num_threads=4'.format(sfm_exec, fullpath)
    print command
    subprocess.call(command, shell=True)
    command = 'rm -rf {}/mvg'.format(fullpath)
    print command
    subprocess.call(command, shell=True)
    

