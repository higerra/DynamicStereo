import subprocess

datasets = ['vegas32', 'vegas33']
testFrame = ['00100','00200']

root_path = '~/Documents/research/DynamicStereo/data'
stereo_path = '~/Documents/research/DynamicStereo/code/build/DynamicStereo/DynamicStereo'

for data in datasets:
    for tf in testFrame:
        command = '{} {}/data_{} --testFrame={} --resolution=128'.format(stereo_path, root_path, data, tf)
        print command
        subprocess.check_call(command, shell=True)    
