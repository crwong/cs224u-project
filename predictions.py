#!/usr/bin/env python

# import bagged
import glove
# import tfidf_parse

import numpy as np

SUFFIX = 'tiny'
TRAIN_FILE = 'data/topics_%s/ALL_CLEAN_%s.txt' % (SUFFIX, SUFFIX)
MODEL = 'gloveIMDB'
OUTPUT = 'ensemble/out_%s-%s.txt' % (MODEL, SUFFIX)

def main():
  # model, trainMat, trainVals, cutoff = tfidf_parse.get_tfidf_logreg(TRAIN_FILE)
  # testMat = trainMat[cutoff:]
  # testVals = trainVals[cutoff:]

  model, trainMat, trainVals = glove.get_glove_logreg(TRAIN_FILE)
  testMat = trainMat[(trainMat.shape[0]*0.7):,:]
  testVals = trainVals[(trainMat.shape[0]*0.7):]

  np.savetxt(OUTPUT, model.predict_proba(testMat))
  print model.score(testMat, testVals)


if __name__ == "__main__":
  main()
