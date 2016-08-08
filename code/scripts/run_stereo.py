import subprocess
import argparse
from os.path import dirname

stereo_path = '~/Documents/research/DynamicStereo/code/build/DynamicStereo/DynamicStereo'

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
    command = "{} {}/{} --testFrame={}".format(stereo_path, path, dataset, testFrame)
    print command
    subprocess.call(command, shell=True)
