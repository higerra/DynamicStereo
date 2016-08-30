import subprocess
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--input', default='/home/yanhang/Documents/research/DynamicStereo/data/working/list_stereo.txt')
parser.add_argument('--output', default='/home/yanhang/Documents/research/DynamicStereo/data/working/prewarp')

data_root = '/home/yanhang/Documents/research/DynamicStereo/data/working'
exec_path = '/home/yanhang/Documents/research/DynamicStereo/code/build/VideoSegmentation/VideoSegmentation'

args = parser.parse_args()
listfile = open(args.input)

datasets = listfile.readlines()

for dataset in datasets:
    info = dataset.split()
    assert(len(info) == 2)
    name = info[0]
    testframe = info[1]

    command = 'ffmpeg -y -i {}/{}/midres/prewarp/prewarpb{:05d}_%05d.jpg -vcodec h264 -qp 0 {}/prewarp_{}.mp4'\
        .format(data_root, name, int(testframe), args.output, name)
    print command
    subprocess.check_call(command, shell=True)

    # command = '{} {}/prewarp_{}.mp4 {}'.format(exec_path, args.output, name, args.output)
    # print command
    # subprocess.check_call(command, shell=True)
