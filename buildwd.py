import numpy as np
from collections import defaultdict

def processWord(word):
    return (word.translate(None, '!.?!,\"\'\\')).lower()

"""
File should have a tweet on each line. Each line should contain id, subject, tweet.
"""
def buildWords(file_name):
    wordCountDict = defaultdict(int)
    wordRowDict = {}
    numTweets = 0
    f = open(file_name)
    row = 0
    for line in f:
        numTweets += 1
        words = line.split()
        for word in words[2:]:
            pword = processWord(word)
            wordCountDict[pword] = wordCountDict[pword] + 1
            if wordCountDict[pword] > 10 and pword not in wordRowDict:
                wordRowDict[pword] = row
                row += 1
    f.close()
    print len(wordRowDict)
    return wordRowDict, numTweets

"""
Doesn't work well.
"""
def writeToCSV(mat, colnames, wordRowDict, file_name):
    f = open(file_name, 'w')
    toWrite = "Words, "
    for col in colnames:
        toWrite += col
        if not col == colnames[-1]:
            toWrite += ", "
    f.write(toWrite)
    for word in wordRowDict:
        print word
        toWrite = word + ", "
        for j in range(len(colnames)):
            toWrite += str(mat[wordRowDict[word]][j])
            if not j == len(colnames)-1:
                toWrite += ", "
        f.write(toWrite)
    f.close()  

"""
File should have a tweet on each line. Each line should contain id, subject, tweet.
"""
def buildWD(file_name, writeCSV=False):
    print "Building word dictionary"
    wordRowDict, numTweets = buildWords(file_name)
    print "Word dictionary finished"
    mat = np.zeros((len(wordRowDict), numTweets))
    colnames = []
    subjects = []
    print "Building word document matrix"
    f = open(file_name)
    for line in enumerate(f):
        words = line[1].split()
        colnames.append(words[0])
        subjects.append(words[1])
        tweetColumn = np.zeros(len(wordRowDict))
        for word in words[2:]:
            pword = processWord(word)
            if pword in wordRowDict:
                tweetColumn[wordRowDict[pword]] = tweetColumn[wordRowDict[pword]] + 1
        mat[:,line[0]] = tweetColumn
    f.close()
    print "Word document matrix finished"
    if writeCSV:
        print "Writing to CSV"
        writeToCSV(mat, colnames, wordRowDict, "trainWords.csv")
        print "Finished writing to CSV"
    return (mat, colnames, subjects)
