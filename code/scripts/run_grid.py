import subprocess
import multiprocessing as mp

exec_path = '~/Documents/research/DynamicStereo/code/build/VisualWord/VisualWord'
data_path = '~/Documents/research/DynamicStereo/data/traindata'
output_path = '~/Documents/research/DynamicStereo/data/traindata/gridsearch'

#grid paramters
test_feature = ['hog3d', 'color3d']

#visual word
kCluster = [50, 100, 200, 400]
print 'Extracting features...'
for cluster in kCluster:
    for feature in test_feature:
        save_path = '{}/train_cluster{:05d}_{}.csv'.format(output_path, cluster, feature)
        command = '{} --mode=extract --desc={} --kCluster={} --cache={} {}/samples/list_train.txt'\
            .format(exec_path, feature, cluster, save_path, data_path)
        print command
        save_path = '{}/validation_cluster{:05d}_{}.csv'.format(output_path, cluster, feature)
        subprocess.call(command, shell=True)
        command = '{} --mode=extract --desc={} --kCluster={} --cache={} {}/samples/list_train.txt' \
            .format(exec_path, feature, cluster, save_path, data_path)
        print command
        subprocess.call(command, shell=True)


#for tree classifier
treeClassifier = ['rf', 'bt']
treeDepth = [20, 30, 40]
numTree = [30, 50, 100, 150]

#for svm
kernelType = ['rbf', 'chi2']
paramC = [0.1, 0.5, 1.0, 2.0, 4.0, 8.0]
paramGamma = [0.01, 0.1, 1.0, 2.0, 4.0]

bestParam = []
bestClassifier = ''