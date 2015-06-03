from collections import defaultdict
import numpy as np
LOCATION = 'data/topics_small/'

def processWord(word):
    return (word.translate(None, '!.?!,\"\'\\')).lower()    

def processWords(text):
    words = text.split()
    processedWords = []
    for word in words:
        if not word[0] == '#':
            processedWords.append(processWord(word))
    return processedWords

def buildWords(train_files):
    wordCounts = defaultdict(int)
    wordRows = {}
    row = 0
    numTweets = 0
    for tf in train_files:
        f = open(LOCATION + tf)
        firstLine = True
        for line in f:
            if firstLine:
                firstLine = False
                continue
            numTweets += 1
            text = line.split(',')[0]
            words = processWords(text)
            for word in words:
                wordCounts[word] = wordCounts[word] + 1
                if wordCounts[word] > 70 and word not in wordRows:
                    wordRows[word] = row
                    row += 1
        f.close()
    print len(wordRows)
    return wordRows, numTweets
                

def buildDataset(train_files):
    wordRows, numTweets = buildWords(train_files)
    mat = np.zeros((numTweets, len(wordRows)))
    vals = np.zeros(numTweets)
    label = 1
    index = 0
    print "Building dataset"
    for tf in train_files:
        f = open(LOCATION + tf)
        firstLine = True
        for line in f:
            if firstLine:
                firstLine = False
                continue
            text = line.split(',')[0]
            words = processWords(text)
            row = np.zeros(len(wordRows))
            for word in words:
                if word in wordRows:
                    row[wordRows[word]] = row[wordRows[word]] + 1
            if np.sum(row) > 0 and np.sum(row) > 0.5*(len(words)-2):
                mat[index,:] = row
                vals[index] = label
                index += 1
        f.close()
        label += 1
    mat = mat[0:index,:]
    vals = vals[0:index]
    print "Finished building dataset"
    return (mat, vals)
