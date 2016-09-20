import subprocess

exec_path = '/home/yanhang/Documents/research/DynamicStereo/code/build/VisualWord/VisualWord'
root_dir = '/home/yanhang/Documents/research/DynamicStereo/data/traindata/samples'
model_path = '/home/yanhang/Documents/research/DynamicStereo/data/traindata/model_color3d_cluster00050.rf'
codebook_path = '/home/yanhang/Documents/research/DynamicStereo/data/traindata/codebook_color3d_cluster00050.txt'

testset = ['sample001.mp4', 'sample002.mp4', 'sample006.mp4', 'sample007.mp4', 'sample014.mp4', 'sample015.mp4',
           'sample034.mp4', 'sample035.mp4', 'sample038.mp4', 'sample039.mp4', 'sample053.mp4', 'sample054.mp4']

for test in testset:
    command = '{} --mode=detect --desc=color3d --model={} --codebook={} {}/{} ' \
              '{}/segmentation/{}.pb'.format(exec_path, model_path, codebook_path, root_dir, test, root_dir, test)
    print command
    subprocess.call(command, shell=True)

