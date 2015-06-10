import numpy as np
from collections import defaultdict
import random

def processWord(word):
    return (word.translate(None, '!.?!,\"\'\\')).lower()

"""
File should have a tweet on each line. Each line should contain id, subject, tweet.
"""
def buildWords(file_name):
    wordCountDict = defaultdict(int)
    wordRowDict = {}
    rownames = []
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
                rownames.append(pword)
                wordRowDict[pword] = row
                row += 1
    f.close()
    print len(wordRowDict)
    return wordRowDict, numTweets, rownames

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
Builds the WD matrix.
colnames contains the tweet ids.
rownames contains the words.
subjects contains the correct subjects in the order of colnames.
File should have a tweet on each line. Each line should contain id, subject, tweet.
"""
def buildWD(file_name, writeCSV=False):
    print "Building word dictionary"
    wordRowDict, numTweets, rownames = buildWords(file_name)
    print "Word dictionary finished"
    mat = np.zeros((len(wordRowDict), numTweets))
    colnames = []
    subjects = []
    print "Building word document matrix"
    f = open(file_name)
    matCol = 0
    for line in f:
        words = line.split()
        tweetColumn = np.zeros(len(wordRowDict))
        for word in words[2:]:
            pword = processWord(word)
            if pword in wordRowDict:
                tweetColumn[wordRowDict[pword]] = tweetColumn[wordRowDict[pword]] + 1
        if np.sum(tweetColumn) > 0.5*(len(words)-2):
            colnames.append(words[0])
            subjects.append(words[1])
            mat[:,matCol] = tweetColumn
            matCol += 1
    f.close()
    mat = mat[:,0:matCol]
    print "Word document matrix finished"
    if writeCSV:
        print "Writing to CSV"
        writeToCSV(mat, colnames, wordRowDict, "trainWords.csv")
        print "Finished writing to CSV"

    # RANDOMIZE
    shuffle = range(len(subjects))
    random.shuffle(shuffle)
    m = np.zeros(mat.shape)
    c = []
    s = []
    index = 0
    for i in shuffle:
        m[:, index] = mat[:, i]
        c.append(colnames[i])
        s.append(subjects[i])
        index += 1

    return (m, c, rownames, s)
    #return (mat, colnames, rownames, subjects)


subjectToValue = ['android','basic','coffee','dontjudgeme','earthquake','egypt',
    'election','freedom','god','haiti','happy','harrypotter','healthcare',
    'immigration','indonesia','ipod','love','mubarak','obama','obamacare',
    'question','sotu','teaparty','tsunami','usa','win','wiunion']

subjectToValue_sports_politics = ['Politics', 'Sports']

def trainValsFromSubjects(subjects, sports_politics_dataset=False):
    reference = subjectToValue
    if sports_politics_dataset:
        reference = subjectToValue_sports_politics

    trainVals = np.zeros(len(subjects))
    for s in enumerate(subjects):
        trainVals[s[0]] = reference.index(s[1])
    return trainVals

