#!/usr/bin/env python

SUFFIX = 'small'
INPUT = 'topics_%s/ALL_CLEAN_%s.txt' % (SUFFIX, SUFFIX)
OUTPUT = 'topics_%s/ALL_TEXT_%s.txt' % (SUFFIX, SUFFIX)

def main():
  infile = open(INPUT, 'r')
  outfile = open(OUTPUT, 'w+')
  for line in infile:
    s = line[line.find(' ', line.find(' ') + 1):].strip()
    outfile.write('%s\n' % s)
  infile.close()
  outfile.close()

if __name__ == "__main__":
  main()
