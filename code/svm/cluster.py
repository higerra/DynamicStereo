import cv2
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from matplotlib import pyplot as plt

import argparse

parser = argparse.ArgumentParser(prog='cluster', description='Cluster the time sequence')
parser.add_argument('input')

max_frame = 50
downsample = 4

args = parser.parse_args()

cap = cv2.VideoCapture(args.input)
assert(cap.isOpened())

kFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) / downsample
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) / downsample
assert(kFrames >= max_frame)

print "Done. kFrame:{}, width:{}, height:{}".format(kFrames, width, height)
samples = np.zeros((width*height, max_frame*3), dtype=float)
print 'Reading video...'
fCount = 0
while True:
    ret, img = cap.read()
    if (not ret) or fCount == max_frame:
        break
    img = cv2.resize(img, (width, height),0,0,cv2.INTER_CUBIC)
    samples[:, fCount*3:(fCount*3+3)] = img.reshape(-1, 3) / 255.0
    fCount += 1
print "Done"
cap.release()

print "Meanshift clustering..."


#print "Estimating bandwidth..."
#bw = estimate_bandwidth(samples, quantile=0.1, n_samples=height * width / 10)
bw = 2.0
print "Done. bw:{}".format(bw)

ms = MeanShift(bandwidth=bw)

print "Clustering..."
ms.fit(samples)
labels = ms.labels_
n_cluster = len(np.unique(labels))
print "Done. Number of clusters: {}".format(n_cluster)

colorTable = np.random.rand(n_cluster, 3)
resultVis = np.zeros(height, width, 3)
for y in range(0,height):
    for x in range(0, width):
        resultVis[y, x, :] = colorTable[labels[y*width+x], :]