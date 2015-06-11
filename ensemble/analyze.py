#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

NUM_MODELS = 5
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

def plot():
  plt.figure(1)
  x = [.1 * i for i in range(11)]
  y = [0.62, 0.64373, 0.64, 0.635, 0.62, 0.625, 0.624, 0.603, 0.605, 0.591, 0.592]
  y2 = [0.64373 for i in range(11)]
  plt.title('SENT: Classification Accuracy vs. $\\delta$')
  plt.xlabel('Weight $\\delta$')
  plt.ylabel('Classification Accuracy')
  plt.axis([0,1,.59, .65])
  plt.plot(x,y,'go-',x,y2,'r--')
  plt.savefig('ensemble_sent.png')

  plt.figure(2)
  x = [.1 * i for i in range(11)]
  y = [0.627, 0.628, 0.6355, 0.639, 0.64, 0.64373, 0.642, 0.6423, 0.6401, 0.639, 0.6395]
  plt.title('FREQ: Classification Accuracy vs. $\\alpha$')
  plt.xlabel('Weight $\\alpha$')
  plt.ylabel('Classification Accuracy')
  plt.axis([0,1,.59, .65])

  plt.plot(x,y,'bo-',x,y2,'r--')
  plt.savefig('ensemble_freq.png')

if __name__ == "__main__":
  analyzeOutput()
  plot()
