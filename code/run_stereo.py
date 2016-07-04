import subprocess

datasets = ['casino2', 'casino3', 'casino4', 'casino5', 'casino6', 'casino7', 'casino8', 'statue1', 'shopping1', 'shopping2', 'vegas29', 'vegas32', 'vegas33']
testFrame = ['00100','00200']

root_path = '~/Documents/research/DynamicStereo/data'
stereo_path = '~/Documents/research/DynamicStereo/code/build/DynamicStereo/DynamicStereo'

for data in datasets:
    for tf in testFrame:
        command = '{} {}/data_{} --testFrame={} --resolution=128'.format(stereo_path, root_path, data, tf)
        print command
        subprocess.check_call(command, shell=True)    
