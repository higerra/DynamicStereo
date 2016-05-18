import cv2
import numpy as np
import scipy.misc
import subprocess
from sklearn.datasets import dump_svmlight_file
import argparse

parser = argparse.ArgumentParser(prog='predict')
parser.add_argument('--model', '-m', required=True)
parser.add_argument('--input', '-i', required=True)
parser.add_argument('--output', '-o', default='result.png')
parser.add_argument('--downsample', '-d', default=8)

args = parser.parse_args()

tWindow = 100

cap = cv2.VideoCapture()
cap.open(args.input)
assert(cap.isOpened())

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
kFrame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
assert(kFrame >= tWindow)

width /= args.downsample
height /= args.downsample

pixs = np.zeros((height, width, tWindow * 3))

print 'Extracing feature...'
for i in range(0, tWindow):
    ret, curimg = cap.read()
    assert(curimg is not None)
    pixs[:, :, 3*i:3*(i+1)] = cv2.resize(curimg, (width, height))

features = []
for y in range(0, height):
    for x in range(0, width):
        features.append(pixs[y, x, :]-np.mean(pixs[y, x, :]))


tempTestFile = open('tempTest.txt', 'w')
dump_svmlight_file(features, np.ones(height * width), tempTestFile)

print 'Running svm-predict...'
subprocess.check_call('svm-predict tempTest.txt {} tempLabel.txt'.format(args.model), shell=True)

print 'Read back and visualize...'
outputVis = np.loadtxt(open('tempLabel.txt'))
assert(outputVis is not None)
assert(outputVis.shape[0] == width * height)

outputVis = np.reshape(outputVis, (height, width))
outputVis[outputVis<0] = 0
outputVis[outputVis>0] = 255

subprocess.call('rm tempLabel.txt tempTest.txt', shell=True)

