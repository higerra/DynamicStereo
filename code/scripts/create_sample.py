import subprocess
import argparse
from os.path import dirname

parser = argparse.ArgumentParser()
parser.add_argument('list')
parser.add_argument('startid', type=int)

args = parser.parse_args()
listfile = open(args.list)

size = (640, 360)

samples = listfile.readlines()
path = dirname(args.list)
print path

index = args.startid
for sample in samples:
    name = sample.split()[0]
    startTime = sample.split()[1]
    endTime = sample.split()[2]
    
    command = 'ffmpeg -i {}/{} -vf scale={}:{} -b:v 40000k -ss {} -to {} -an {}/sample{:03d}.mp4'.format(path, name, size[0], size[1], startTime, endTime, path, index)
    print command
    subprocess.call(command, shell=True)

    command = 'ffmpeg -i {}/{} -ss {} -vframes 1 {}/snap{:03d}.png'.format(path, name, startTime, path, index)
    print command
    subprocess.call(command, shell=True)
    
    index = index + 1

    
listfile.close()
