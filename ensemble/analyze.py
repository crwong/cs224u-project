#!/usr/bin/env python

import numpy as np

NUM_MODELS = 3
OUTPUT = 'ensemble_out.txt'

def analyzeOutput():
  infile = open(OUTPUT, 'r')
  weights = []
  topScore = 0.0
  for line in infile:
    l = line.split(',')
    l = [float(n) for n in l]
    if topScore < l[NUM_MODELS]:
      topScore = l[NUM_MODELS]
      weights = l[0:NUM_MODELS]
  infile.close()
  print topScore, weights

if __name__ == "__main__":
  analyzeOutput()
