import cv2
import numpy as np
from sklearn.datasets import dump_svmlight_file

tWindow = 100
prefix = "../../data/svmTrain/"


downsample = 4
negative_stride = 4

index = 1

features = []
labels = []

train_all = open(prefix+'/train_all.txt', 'a')

max_sample = 2

while True:
    print "Processing sample {}".format(index)
    gtimg = cv2.imread('{}/samples/gt{}.png'.format(prefix, index), cv2.IMREAD_GRAYSCALE)
    if gtimg is None:
        break

    height,width = gtimg.shape
    height /= downsample
    width /= downsample
    gtimg = cv2.resize(gtimg, (width, height),interpolation=cv2.INTER_NEAREST)

    print "frame size: {},{}".format(int(width), int(height))
    cap = cv2.VideoCapture('{}/samples/sample{}.mp4'.format(prefix, index))
    assert(not cap is None)

    kFrame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert(kFrame >= tWindow)

    print "Reading video..."
    pixs = np.zeros((height, width, kFrame * 3), dtype=int)
    for i in range(0,kFrame):
        ret, curimg = cap.read()
        pixs[:,:,3*i:3*(i+1)] = cv2.resize(curimg, (width, height))

    #normalization
    print "Extracing sample..."
    posCount = 0.0
    negCount = 0.0
    for y in range(0, height):
        for x in range(0, width):
            pixs[y,x,:] = pixs[y,x,:] - np.mean(pixs[y,x,:])
            if gtimg[y,x] > 200:
                for i in range(0, kFrame-tWindow, tWindow/2):
                    features.append(pixs[y,x,i*3:(i+tWindow)*3])
                    labels.append(1)
                    posCount += 1
            elif gtimg[y,x] < 20:
                if x % negative_stride == 0 and y % negative_stride == 0:
                    for i in range(0, kFrame - tWindow, tWindow / 2):
                        features.append(pixs[y, x, i * 3:(i + tWindow) * 3])
                        labels.append(-1)
                        negCount += 1

    print "Saving, total number of samples:{}, ratio of positive:{:.2f}, ratio of negative:{:.2f}...".format(len(features), posCount/(posCount+negCount), negCount/(posCount+negCount))
    traindata = open(prefix + '/train{}.txt'.format(index), 'w')
    dump_svmlight_file(features, labels, traindata)
    dump_svmlight_file(features, labels, train_all)

    index += 1
    if index > max_sample:
        break

print "All done"