#!/usr/bin/env python

import bagged
import glove
import tfidf_parse

import numpy as np

SUFFIX = 'micro'
TRAIN_FILE = 'data/topics_%s/ALL_CLEAN_%s.txt' % (SUFFIX, SUFFIX)
MODEL = 'bagged'
OUTPUT = 'out_%s-%s.txt' % (MODEL, SUFFIX)

def main():
  model, trainMat, trainVals = bagged.get_bag_logreg(TRAIN_FILE)

  testMat = trainMat[(trainMat.shape[0]*0.7):,:]
  testVals = trainVals[(trainMat.shape[0]*0.7):]

  np.savetxt(OUTPUT, model.predict_proba(testMat))


if __name__ == "__main__":
  main()
