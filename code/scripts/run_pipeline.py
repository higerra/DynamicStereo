import subprocess
import argparse

stereo_exec = '/home/yanhang/Documents/research/DynamicStereo/code/build/DynamicStereo/DynamicStereo'
render_exec = '/home/yanhang/Documents/research/DynamicStereo/code/build/SegmentAndRender/SegmentAndRender'

data_path = '/home/yanhang/Documents/research/DynamicStereo/data/working/data_newyork2'
startid = 50
endid = 250
interval = 20

for tf in range(startid, endid, interval):
    command = 'rm {}/midres/depth{:05d}.depth'.format(data_path, tf);
    print command
    subprocess.call(command, shell=True)

    command = '{} {} --testFrame=tf'.format(stereo_exec, tf)
    print command
    subprocess.call(command, shell=True)

    command = '{} {} --testFrame=tf'.format(render_exec, tf)
    print command
    subprocess.call(command, shell=True)
    
    
