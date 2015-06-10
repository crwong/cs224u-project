#!/usr/bin/env python

import numpy as np

MODELS = [
  'bagged',
  'gloveTwitter',
  'tfidf',
  'baggedSent',
  'gloveIMDB'
]
NUM_MODELS = len(MODELS)
FILES = [('out_%s-tiny.txt' % model) for model in MODELS]
TRUTHS = np.loadtxt('tiny-values.txt')
WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
OUTPUT = 'ensemble_out.txt'

def calculateScore(predictions):
  count = 0
  for i in xrange(len(TRUTHS)):
    if predictions[i] == TRUTHS[i]: count += 1
  return float(count) / len(TRUTHS)

def output(s, outfile):
  print s
  outfile.write(s + '\n')

def ensemble(models, weights, outfile):
  predictions = []
  for i in xrange(len(models[0])):
    arr = weights[0] * models[0][i] +\
          weights[1] * models[1][i] +\
          weights[2] * models[2][i] +\
          weights[3] * models[3][i] +\
          weights[4] * models[4][i]
    prediction = np.argmax(arr)
    predictions.append(prediction)
  assert len(predictions) == len(TRUTHS)
  score = calculateScore(predictions)
  s = '%.1f,%.1f,%.1f,%.1f,%.1f,%.5f' %\
      (weights[0], weights[1], weights[2], weights[3], weights[4], score)
  output(s, outfile)

def main():
  models = [np.loadtxt(f) for f in FILES]
  outfile = open(OUTPUT, 'w+')
  for weight0 in WEIGHTS:
    for weight1 in WEIGHTS:
      for weight2 in WEIGHTS:
        for weight3 in WEIGHTS:
          for weight4 in WEIGHTS:
            weights = [weight0, weight1, weight2, weight3, weight4]
            ensemble(models, weights, outfile)
  outfile.close()

if __name__ == "__main__":
  main()
