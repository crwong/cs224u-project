import sentiment_buildwd
import numpy as np
import tweetprocess
import random
from sklearn import linear_model

TRAIN_FILE = 'data/sentiment/training.1600000.processed.noemoticon.csv'

def tfidf_logreg(train_file):
    wd = sentiment_buildwd.buildWD(train_file)
    colnames = wd[1]
    rownames = wd[2]
    subjects = wd[3]

    trainMat = np.transpose(wd[0])
    row_sums = trainMat.sum(axis=1)
    trainMat = trainMat / row_sums[:, np.newaxis]

    trainVals = sentiment_buildwd.trainValsFromSubjects(subjects)

    # RANDOMIZE
    random.seed(17)
    shuffle = range(len(subjects))
    random.shuffle(shuffle)
    train = []
    labels = []
    index = 0
    for i in shuffle:
        train.append(trainMat[i])
        labels.append(trainVals[i])
        index += 1
    cutoff = int(index*0.7)

    logreg = linear_model.LogisticRegression()
    logreg.fit(train[0:cutoff], labels[0:cutoff])
    return logreg.score(train[cutoff:], labels[cutoff:])


if __name__ == "__main__":
    score_logreg = tfidf_logreg(TRAIN_FILE)
    print 'LogReg: ', score_logreg
