import subprocess
import argparse
from os.path import dirname

parser = argparse.ArgumentParser()
parser.add_argument('input')

args = parser.parse_args()
listfile = open(args.input)

seg_binary_dir = "~/Documents/research/external_code/video_segment/seg_tree_sample/build/seg_tree_sample"

samples = listfile.readlines()
path = dirname(args.input)
print path

for sample in samples:
    name = sample.split()[0]
    command = "{} --input_file={}/{} --write_to_file --render_and_save".format(seg_binary_dir, path, name)
    print command
    subprocess.call(command, shell=True)
