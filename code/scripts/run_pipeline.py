import subprocess
import argparse
import json
import os.path

parser = argparse.ArgumentParser()
parser.add_argument('list', default='/home/yanhang/Documents/research/DynamicStereo/data/working/list_all.txt')
parser.add_argument('--clean_stereo', action='store_true', help='re-run stereo', default=False)
parser.add_argument('--clean_segmentation', action='store_true', help='re-run segmentation from begining', default=False)
parser.add_argument('--clean_classification', action='store_true', help='re-run classification')
parser.add_argument('--skip_stereo', action='store_true', default=False)
parser.add_argument('--skip_render', action='store_true', default=False)

args = parser.parse_args()

listfile = open(args.list)
datasets = listfile.readlines()

stereo_exec = '/home/yanhang/Documents/research/DynamicStereo/code/build/DynamicStereo/DynamicStereo'
render_exec = '/home/yanhang/Documents/research/DynamicStereo/code/build/SegmentAndRender/SegmentAndRender'
data_root = '/home/yanhang/Documents/research/DynamicStereo/data/working/'

for data in datasets:
    info = data.split()
    name = info[0]

    data_path = data_root + name + '/'

    # if dataset specific configuration doesn't exist, use default configuration
    if os.path.isfile(data_path + '/conf.json'):
        with open(data_path + '/conf.json') as f:
            conf = json.load(f)
    else:
        with open(data_root + '/default.json') as f:
            conf = json.load(f)

    global_stereo = ''
    global_render = ''
    if 'global_stereo' in conf:
        global_stereo += conf['global_stereo']
    if 'global_render' in conf:
        global_render += conf['global_render']

    for frame in conf['frames']:
        tf = frame['frameid']
        if args.clean_stereo:
            command = 'rm {}/midres/depth{:05d}.depth'.format(data_path, tf)
            print command
            subprocess.call(command, shell=True)
            args.clean_segmentation = True

        if args.clean_segmentation:
            command = 'rm {}/midres/segment{:05d}.yml'.format(data_path, tf)
            print command
            subprocess.call(command, shell=True)
            args.clean_classification = True

        if args.clean_classification:
            command = 'rm {}/midres/classification{:05d}.png'.format(data_path, tf)
            print command
            subprocess.call(command, shell=True)

        if not args.skip_stereo:
            command = '{} {} --testFrame={}'.format(stereo_exec, data_path, tf)
            if 'weight_smooth' in frame:
                command += ' weight_smooth={}'.format(frame['weight_smooth'])
            command += ' ' + global_stereo
            print command
            subprocess.call(command, shell=True)

        if not args.skip_render:
            command = '{} {} --testFrame={}'.format(render_exec, data_path, tf)
            command += ' ' + global_render
            print command
            subprocess.call(command, shell=True)
