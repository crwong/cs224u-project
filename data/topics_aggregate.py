#!/usr/bin/env python

import os
import re

DIR_NAME = 'topics_micro'
SUFFIX = '_micro.txt'
OUTPUT = 'ALL' + SUFFIX
OUTPUT_CLEAN = 'ALL_CLEAN' + SUFFIX
OUTPUT_TAGS = 'ALL_TAGS' + SUFFIX

# INPUTS
# text,to_user_id,from_user,id,from_user_id,iso_language_code,
# source,profile_image_url,geo_type,geo_coordinates_0,
# geo_coordinates_1,created_at,time

# OUTPUT
# id topic text

# OUTPUT TAGS
# id tags(space-separated)

# General hashtags
r1 = re.compile('(#.*?) ')
# Hashtag at end of tweet
r2 = re.compile('(#.*?)$')

def write_clean_file(tweet_id, topic, text, outfile_clean, outfile_tags):
  # Remove RT
  text = re.sub('RT', '', text)

  # Filter hashtags
  tags = r1.findall(text)
  text = r1.sub('', text)
  tags += r2.findall(text)
  text = r2.sub('', text)

  outfile_clean.write('%s %s %s\n' % (tweet_id, topic, text))
  outfile_tags.write('%s %s\n' % (tweet_id, ' '.join(tags)))

def aggregate_file(filename, outfile, outfile_clean, outfile_tags):
  print 'Aggregating', filename
  topic = filename[:filename.find(SUFFIX)]
  infile = open(os.path.join(DIR_NAME, filename), 'r')
  infile.readline()
  for line in infile:
    arr = line.split(',')
    assert len(arr) == 13
    text = arr[0]
    tweet_id = arr[3]
    outfile.write('%s %s %s\n' % (tweet_id, topic, text))
    write_clean_file(tweet_id, topic, text, outfile_clean, outfile_tags)
  infile.close()

if __name__ == "__main__":
  target_filename = os.path.join(DIR_NAME, OUTPUT)
  target_filename_clean = os.path.join(DIR_NAME, OUTPUT_CLEAN)
  target_filename_tags = os.path.join(DIR_NAME, OUTPUT_TAGS)
  outfile = open(target_filename, 'w+')
  outfile_clean = open(target_filename_clean, 'w+')
  outfile_tags = open(target_filename_tags, 'w+')
  for f in os.listdir(DIR_NAME):
    if f == OUTPUT or f == OUTPUT_CLEAN or f == OUTPUT_TAGS: continue
    aggregate_file(f, outfile, outfile_clean, outfile_tags)
  outfile.close()
  outfile_clean.close()
  outfile_tags.close()
  print 'DONE'
