import subprocess
import argparse
from os.path import dirname

exec_path = '/home/yanhang/Documents/research/DynamicStereo/code/build/SegmentAndRender/SegmentAndRender'
model_path = '/home/yanhang/Documents/research/DynamicStereo/data/traindata/visualword/model_new.rf'
metainfo_path = '/home/yanhang/Documents/research/DynamicStereo/data/traindata/visualword/metainfo_new_cluster00050.yml'
data_root = '/home/yanhang/Documents/research/DynamicStereo/data/working'

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='/home/yanhang/Documents/research/DynamicStereo/data/working/list_stereo.txt')
parser.add_argument('--output', default='/home/yanhang/Documents/research/DynamicStereo/data/working/result')
parser.add_argument('--recompute', action='store_true', help='recompute segmentation')
args = parser.parse_args()

listfile = open(args.input)

datasets = listfile.readlines()

regularization = ['median', 'RPCA', 'anisotropic', 'poisson']

for dataset in datasets:
    info = dataset.split()
    assert(len(info) == 2)
    name = info[0]
    testFrame = info[1]

    #command = 'rm {}/{}/midres/classification*'.format(data_root, name)
    #print command
    #subprocess.call(command, shell=True)

    if args.recompute:
        command = 'rm {}/{}/midres/segment*.yml'.format(data_root, name)
        print command
        subprocess.call(command, shell=True)

    #run with different rendering algorithm
    for reg in regularization:
        command = '{} {}/{} --testFrame={} --classifierPath={} --codebookPath={} --regularization={}'\
            .format(exec_path, data_root, name, testFrame, model_path, metainfo_path, reg)
        print command
        subprocess.check_call(command, shell=True)

        command = 'ffmpeg -y -i {}/{}/temp/regulared_{}_{:05d}_%05d.jpg -vcodec h264 -qp 0 {}/{}/temp/regulared_{}.mp4'\
            .format(data_root, name, reg, int(testFrame), data_root, name, reg)
        print command
        subprocess.check_call(command, shell=True)

