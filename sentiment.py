import sentiment_buildwd
import numpy as np
import tweetprocess
from sklearn import linear_model

TRAIN_FILE = 'data/sentiment/training.1600000.processed.noemoticon.csv'

def tfidf(mat=None, rownames=None):
    """TF-IDF on mat. rownames is unused; it's an argument only
    for consistency with other methods used here"""
    colsums = np.sum(mat, axis=0)
    doccount = mat.shape[1]
    w = np.array([_tfidf_row_func(row, colsums, doccount) for row in mat])
    return (w, rownames)

def _tfidf_row_func(row, colsums, doccount):
    df = float(len([x for x in row if x > 0]))
    idf = 0.0
    # This ensures a defined IDF value >= 0.0:
    if df > 0.0 and df != doccount:
        idf = np.log(doccount / df)
    tfs = row/colsums
    return tfs * idf

def tfidf_logreg(train_file):
    wd = sentiment_buildwd.buildWD(train_file)
    colnames = wd[1]
    rownames = wd[2]
    subjects = wd[3]
    idf = tfidf(wd[0], rownames)

    trainMat = np.zeros((len(colnames), wd[0].shape[1]))
    f = open(train_file)
    matCol = 0
    for line in f:
        words = line.strip('\"').split(',')
        if words[1] in colnames:
            trainRow = np.zeros(wd[0].shape[1])
            numWords = 0
            tweet = sentiment_buildwd.buildTweet(words[5:])
            for word in tweetprocess.tokenize(tweet):
                pword = sentiment_buildwd.processWord(word)
                if pword in rownames:
                    numWords += 1
                    trainRow = trainRow + idf[0][rownames.index(pword)]
            trainRow = (trainRow*1.0) / numWords
            trainMat[matCol,:] = trainRow
            matCol += 1
    f.close()

    trainVals = sentiment_buildwd.trainValsFromSubjects(subjects)

    logreg = linear_model.LogisticRegression()
    logreg.fit(trainMat[0:(trainMat.shape[0]*0.7),:], trainVals[0:(trainMat.shape[0]*0.7)])
    return logreg.score(trainMat[(trainMat.shape[0]*0.7):,:], trainVals[(trainMat.shape[0]*0.7):])


if __name__ == "__main__":
    score_logreg = tfidf_logreg(TRAIN_FILE)
    print 'LogReg: ', score_logreg
