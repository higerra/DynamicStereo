from sklearn import svm, grid_search, datasets
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True, help='path to training data')
parser.add_argument('-cv', default=5, type=int)

args = parser.parse_args()
trainData = open(args.input)

assert(trainData is not None)
[features, labels] = datasets.load_svmlight_file(trainData)

params = {'C': [2**-3, 0.5, 1, 2, 4, 8],
          'gamma': [16, 8, 2, 1, 0.5, 2**-3]}

svc = svm.SVC()
print 'Start searching...'

clf = grid_search.GridSearchCV(svc, param_grid=params, n_jobs=4, cv=args.cv, verbose=5)
clf.fit(features, labels)

bestParam = clf.best_params_
print 'best param:'
print bestParam
print 'best score: {}'.format(clf.score(features, labels))