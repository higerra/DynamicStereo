import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('listfile', default='/home/yanhang/Documents/research/DynamicStereo/data/working/list_datasets.txt')
parser.add_argument('--remote_ip', default='yanhang@23.99.216.149')
parser.add_argument('--local_root', default='/home/yanhang/Documents/research/DynamicStereo/data/working/')
parser.add_argument('--remote_root', default='~/hang/data/working/')

args = parser.parse_args()

list_file = open(args.listfile)
datasets = list_file.readlines()

for data in datasets:
    info = data.split()
    if len(info) == 0:
        break
    name = info[0]
    # command = 'scp -r {}:{}/{}/temp {}/{}/'.format(args.source_ip, args.remote_root, name, args.local_root, name)
    # print command
    # subprocess.call(command, shell=True)

    # command = 'scp {}:{}/{}/midres/depth*.depth {}/{}/midres/'.format(args.source_ip, args.remote_root, name,
    #                                                                   args.local_root, name)
    # print command
    # subprocess.call(command, shell=True)

    # command = 'scp {}:{}/{}/midres/classification* {}/{}/midres/'.format(args.source_ip, args.remote_root, name,
    #                                                                   args.local_root, name)
    # print command

    command = 'scp {}/{}/conf.json {}:{}/{}/'.format(args.local_root, name, args.remote_ip, args.remote_root, name)
    subprocess.call(command, shell=True)

    command = 'scp {}/{}/midres/classi* {}:{}/{}/midres/'.format(args.local_root, name, args.remote_ip, args.remote_root, name)
    subprocess.call(command, shell=True)

    # command = 'scp {}:{}/{}/midres/segment*.yml {}/{}/midres/'.format(args.source_ip, args.remote_root, name,
    #                                                                      args.local_root, name)
    # print command
    # subprocess.call(command, shell=True)


