#!/usr/bin/env python

import os

NUM_TWEETS = 50

SUFFIX = '_micro'
DIR_NAME = 'topics'
TARGET_DIR_NAME = 'topics' + SUFFIX
EXT = '.txt'

def extract_file(filename):
  path = os.path.join(DIR_NAME, filename)
  split_index = len(filename) - len(EXT)
  target_filename = filename[0:split_index] + SUFFIX + filename[split_index:]
  target_path = os.path.join(TARGET_DIR_NAME, target_filename)

  infile = open(path, 'r')
  outfile = open(target_path, 'w+')
  for i in xrange(NUM_TWEETS + 1):     # First line is CSV header
    outfile.write(infile.readline())
  infile.close()
  outfile.close()

if __name__ == "__main__":
  for filename in os.listdir(DIR_NAME):
    if filename.endswith(EXT):
      print 'Extracting file:', filename
      extract_file(filename)
  print 'DONE'
