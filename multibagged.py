import multiprocess
import random
import numpy as np
from sklearn import linear_model
TRAIN_FILES = ['android_small.txt', 'basic_small.txt', 'coffee_small.txt',
               'dontjudgeme_small.txt', 'earthquake_small.txt', 'egypt_small.txt',
               'election_small.txt', 'freedom_small.txt', 'god_small.txt',
               'haiti_small.txt', 'happy_small.txt', 'harrypotter_small.txt',
               'healthcare_small.txt', 'immigration_small.txt', 'indonesia_small.txt',
               'ipod_small.txt', 'love_small.txt', 'mubarak_small.txt', 'obama_small.txt',
               'obamacare_small.txt', 'question_small.txt', 'sotu_small.txt',
               'teaparty_small.txt', 'tsunami_small.txt', 'usa_small.txt', 'win_small.txt',
               'wiunion_small.txt']

def bag_logreg():
    mat, vals = multiprocess.buildDataset(TRAIN_FILES)
    row_sums = mat.sum(axis=1)
    mat = mat / row_sums[:, np.newaxis]

    trainSize = 0.7*len(vals)
    trainIndexes = random.sample(range(len(vals)), len(vals))
    trainMat = np.zeros((trainSize, mat.shape[1]))
    trainVals = np.zeros(trainSize)
    
    testMat = np.zeros((len(vals)-trainSize, mat.shape[1]))
    testVals = np.zeros(len(vals)-trainSize)
    print trainMat.shape
    index = 0
    for ti in trainIndexes:
        if index < trainMat.shape[0]:
            trainMat[index, :] = mat[ti, :]
            trainVals[index] = vals[ti]
        elif index - trainMat.shape[0] < testMat.shape[0]:
            testMat[index - trainMat.shape[0], :] = mat[ti, :]
            testVals[index - trainMat.shape[0]] = vals[ti]
        index += 1
    logreg = linear_model.LogisticRegression()
    print "Beginning training of model"
    logreg.fit(trainMat, trainVals)
    print "Finished training of model"
    print logreg.score(trainMat, trainVals)
    return logreg.score(testMat, testVals)

score = bag_logreg()
        
    
