import subprocess
exec_path = '~/Documents/research/DynamicStereo/code/build/VisualWord/VisualWord'
data_path = '~/Documents/research/DynamicStereo/data/traindata'
output_path = '/home/yanhang/Documents/research/DynamicStereo/data/traindata/gridsearch'

#grid paramters
test_feature = ['hog3d', 'color3d']
#test_feature = ['color3d']

# #visual word
kCluster = [50, 100, 200, 500]
print 'Extracting features...'
for cluster in kCluster:
    for feature in test_feature:
        codebook_path = '{}/model_{}_cluster{:05d}_codebook.txt'.format(output_path, feature, cluster)
        save_path = '{}/train_{}'.format(output_path, feature)
        command = '{} --mode=multiExtract --desc={} --cache={} --codebook={} {}/samples/list_train.txt'\
            .format(exec_path, feature, save_path, codebook_path, data_path)
        print command
        save_path = '{}/validation_{}'.format(output_path, feature)
        subprocess.call(command, shell=True)
        command = '{} --mode=multiExtract --desc={} --cache={} --codebook={} {}/samples/list_validation.txt' \
            .format(exec_path, feature, save_path,  codebook_path, data_path)
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

log_file = open('{}/log.txt'.format(output_path), 'w')

bestValidation = -1.0
for classifier in treeClassifier:
    for feature in test_feature:
        for cluster in kCluster:
            for td in treeDepth:
                for nt in numTree:
                    cur_param = [td, nt]
                    print "Classifier: {}, cluster: {}, tree depth: {}, number of tree: {}"\
                        .format(classifier, cluster, td, nt)
                    train_path = '{}/train_{}_cluster{:05d}.csv'.format(output_path, feature, cluster)
                    validation_path = '{}/validation_{}_cluster{:05d}.csv'.format(output_path, feature, cluster)
                    model_path = '{}/model_{}_cluster{:05d}'.format(output_path, feature, cluster)
                    codebook_path = '{}/model_{}_cluster{:05d}_codebook.txt'.format(output_path, feature, cluster)
                    command = "{} --mode=train --cache={} --codebook={} --validation={} --model={} --classifier={} --numTree={} " \
                              "--treeDepth={} null.txt | grep 'Validation'"\
                        .format(exec_path, train_path, codebook_path, validation_path, model_path, classifier, nt, td)
                    print command
                    output = subprocess.check_output(command, shell=True)
                    cur_acc = float(output.split()[-1])
                    print "Validation accuracy: ", cur_acc
                    log_file.write('{}\t{}\tnt:{}\ttd:{}\tvalidation:{:.3f}'
                                   .format(classifier, feature, td, nt, cur_acc))
                    if cur_acc > bestValidation:
                       bestValidation = cur_acc
                       bestClassifier = classifier
                       bestParam = cur_param

#grid search for SVM
for feature in test_feature:
    for cluster in kCluster:
        for c in paramC:
            for gamma in paramGamma:
                print "Classifier: svm, cluster: {}, C: {:.3f}, gamma: {:.3f}".format(cluster, c, gamma)
                cur_param = [c, paramGamma]
                train_path = '{}/train_{}_cluster{:05d}.csv'.format(output_path, feature, cluster)
                validation_path = '{}/validation_{}_cluster{:05d}.csv'.format(output_path, feature, cluster)
                model_path = '{}/model_{}_cluster{:05d}'.format(output_path, feature, cluster)
                codebook_path = '{}/model_{}_cluster{:05d}_codebook.txt'.format(output_path, feature, cluster)
                command = "{} --mode=train --cache={}  codebook_path={} --validation={} --model={} --classifier={} --svmC={}" \
                    " --svmGamma={} null.txt | grep 'Validation'"\
                    .format(exec_path, train_path, codebook_path, validation_path, model_path, 'svm', c, gamma)

                print command
                output = subprocess.check_output(command, shell=True)
                cur_acc = float(output.split()[-1])
                print "Validation accuracy: ", cur_acc

                log_file.write('{}\t{}\tC:{}\tGamma:{}\tvalidation:{:.3f}'
                               .format('svm', feature, c, gamma, cur_acc))
                if cur_acc > bestValidation:
                    bestValidation = cur_acc
                    bestClassifier = 'svm'
                    bestParam = cur_param

log_file.close()

print "All done"
print "Best classifier:", bestClassifier
print "Best parameter: ", bestParam
print "Best validation accuracy:", bestValidation
