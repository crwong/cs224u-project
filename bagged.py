import buildwd
import numpy as np
from sklearn import linear_model
from sklearn import neighbors

TRAIN_FILE = 'data/topics_small/ALL_small.txt'

def bag_logreg(train_file):
    logreg, trainMat, trainVals = get_bag_logreg(train_file)
    return logreg.score(trainMat[(trainMat.shape[0]*0.7):,:], trainVals[(trainMat.shape[0]*0.7):])

def get_bag_logreg(train_file):
    wd = buildwd.buildWD(train_file)
    colnames = wd[1]
    rownames = wd[2]
    subjects = wd[3]

    trainMat = np.transpose(wd[0])
    row_sums = trainMat.sum(axis=1)
    trainMat = trainMat / row_sums[:, np.newaxis]

    trainVals = buildwd.trainValsFromSubjects(subjects)

    print 'Training bag_logreg...'
    logreg = linear_model.LogisticRegression()
    logreg.fit(trainMat[0:(trainMat.shape[0]*0.7),:], trainVals[0:(trainMat.shape[0]*0.7)])
    return logreg, trainMat, trainVals


def bag_knn(train_file):
    knn, trainMat, trainVals = get_bag_knn(train_file)
    return knn.score(trainMat[(trainMat.shape[0]*0.7):,:], trainVals[(trainMat.shape[0]*0.7):])

def get_bag_knn(train_file):
    wd = buildwd.buildWD(train_file)
    colnames = wd[1]
    rownames = wd[2]
    subjects = wd[3]

    trainMat = np.transpose(wd[0])
    row_sums = trainMat.sum(axis=1)
    trainMat = trainMat / row_sums[:, np.newaxis]

    trainVals = buildwd.trainValsFromSubjects(subjects)

    print 'Training bag_knn...'
    knn = neighbors.KNeighborsClassifier(n_neighbors=5)
    knn.fit(trainMat[0:(trainMat.shape[0]*0.7),:], trainVals[0:(trainMat.shape[0]*0.7)])
    return knn, trainMat, trainVals


if __name__ == "__main__":
    # score_knn = bag_knn(TRAIN_FILE)
    # print 'KNN:', score_knn
    score_logreg = bag_logreg(TRAIN_FILE)
    print 'LogReg:', score_logreg

