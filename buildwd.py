import numpy as np
from collections import defaultdict
import sentiment_bagged as sent
import random
import tweetprocess

SENTIMENT_FILENAME = 'data/sentiment/training.1600000.processed.noemoticon.csv'

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
    if "_tiny" in file_name or "_micro" in file_name:
        thresh = 10
    else:
        thresh = 50
    row = 0
    for line in f:
        numTweets += 1
        words = line.split()
        tweet = buildTweet(words[2:])
        for word in tweetprocess.tokenize(tweet):
            wordCountDict[word] = wordCountDict[word] + 1
            if wordCountDict[word] > thresh and word not in wordRowDict:
                rownames.append(word)
                wordRowDict[word] = row
                row += 1
    f.close()
    i = 0
    for word in wordRowDict:
        if i == 15:
            break
        i += 1
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
        toWrite = word + ", "
        for j in range(len(colnames)):
            toWrite += str(mat[wordRowDict[word]][j])
            if not j == len(colnames)-1:
                toWrite += ", "
        f.write(toWrite)
    f.close()

def buildTweet(words):
    tweet = ""
    for i in range(len(words)):
        tweet += words[i]
        if i != len(words)-1:
            tweet += " "
    return tweet

"""
Builds the WD matrix.
colnames contains the tweet ids.
rownames contains the words.
subjects contains the correct subjects in the order of colnames.
File should have a tweet on each line. Each line should contain id, subject, tweet.
"""
def buildWD(file_name, writeCSV=False, randomize=False, sentiment=False):
    print "Building word dictionary"
    wordRowDict, numTweets, rownames = buildWords(file_name)
    print "Word dictionary finished"
    extra_feats = 0
    if sentiment: extra_feats = 1
    mat = np.zeros((len(wordRowDict) + extra_feats, numTweets))
    colnames = []
    subjects = []
    print "Building word document matrix"
    f = open(file_name)
    matCol = 0
    if (sentiment):
        sentiment_model, sentiment_words = sent.tfidf_logreg(SENTIMENT_FILENAME)
        sentimentTweetColumn = np.zeros(len(sentiment_words))
    for line in f:
        words = line.split()
        tweet = buildTweet(words[2:])
        tweetColumn = np.zeros(len(wordRowDict) + extra_feats)
        num_words = 0
        for word in tweetprocess.tokenize(tweet):
            if word in wordRowDict:
                tweetColumn[wordRowDict[word]] = tweetColumn[wordRowDict[word]] + 1
            if sentiment and word in sentiment_words:
                sentimentTweetColumn[sentiment_words.index(word)] += 1
                num_words += 1
        if (sentiment and num_words > 0):
            sentimentTweetColumn *= 1.0 / num_words
            tweetColumn[-1] = sentiment_model.predict(sentimentTweetColumn)
        elif (sentiment):
            tweetColumn[-1] = 2
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

    if randomize:
        random.seed(17)
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

    return (mat, colnames, rownames, subjects)


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

