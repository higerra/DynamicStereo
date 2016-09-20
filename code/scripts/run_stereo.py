import subprocess
import argparse
from os.path import dirname

stereo_path = '~/Documents/research/DynamicStereo/code/build/DynamicStereo/DynamicStereo'
render_path = '~/Documents/research/DynamicStereo/code/build/SegmentAndRender/SegmentAndRender'
classifier_path = '/home/yanhang/Documents/research/DynamicStereo/code/build/VisualWord/temp.rf'
metainfo_path = '/home/yanhang/Documents/research/DynamicStereo/code/build/VisualWord/metainfo_cluster00050.yml'

parser = argparse.ArgumentParser()
parser.add_argument('input')

args = parser.parse_args()
listfile = open(args.input)

samples = listfile.readlines()
path = dirname(args.input)
print path

for sample in samples:
    info = sample.split()
    assert(len(info) == 2)
    dataset = sample.split()[0]

    testFrame = int(sample.split()[1])

    command = "rm {}/{}/midres/depth{:05d}.depth"
    print command
    subprocess.call(command, shell=True)

    command = "{} {}/{} --testFrame={} --logtostderr".format(stereo_path, path, dataset, testFrame)
    print command
    subprocess.call(command, shell=True)

    # command = "ffmpeg -i {}/{}/midres/prewarp/prewarpb{:05d}_%05d.jpg -vcodec h264 -qp 0 -y {}/{}/midres/prewarp/prewarpb{:05d}.mp4".format(
    #     path, dataset, testFrame, path, dataset, testFrame
    # )
    # print command
    # subprocess.call(command, shell=True)
    # command = "cp {}/{}/midres/prewarp/prewarpb{:05d}.mp4 /home/yanhang/Documents/research/DynamicStereo/data/working/prewarp/prewarp_{}{:05d}.mp4".format(
    #     path, dataset, testFrame, dataset, testFrame
    # )
    # print command
    # subprocess.call(command, shell=True)

    # command = '{} {}/{} --testFrame={} --classifierPath={} --codebookPath={}'.format(render_path, path, dataset, testFrame,
    #                                                                                  classifier_path, metainfo_path)
    # print command
    # subprocess.call(command, shell=True)
    #
    # command = 'ffmpeg -i {}/{}/temp/warped{:05d}_%05d.jpg -vcodec h264 -qp 0 -y {}/{}/temp/warped{:05d}.mp4'.format(
    #     path, dataset, testFrame, path, dataset, testFrame
    # )
    # print command
    # subprocess.call(command, shell=True)
    #
    # command = "cp {}/{}/temp/warped{:05d}.mp4 {}/prewarp/warped{}_{:05d}.mp4".format(path, dataset, testFrame,
    #                                                                                path, dataset, testFrame)
    # print command
    # subprocess.call(command, shell=True)

