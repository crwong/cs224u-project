import buildwd
import csv
import numpy as np
import scipy
from operator import itemgetter
from sklearn import linear_model
from sklearn import neighbors
from sklearn import preprocessing

SUFFIX = 'tiny'
TRAIN_FILE = 'data/topics_%s/ALL_CLEAN_%s.txt' % (SUFFIX, SUFFIX)
GLOVE_FILE = 'data/topics_%s/A_GLOVE_%s.txt' % (SUFFIX, SUFFIX)

GLVVEC_LENGTH = 50

GLOVE_CACHE = None

def build(src_filename, delimiter=',', header=True, quoting=csv.QUOTE_MINIMAL):
    reader = csv.reader(file(src_filename), delimiter=delimiter, quoting=quoting)
    colnames = None
    if header:
        colnames = reader.next()
        colnames = colnames[1: ]
    mat = []
    rownames = []
    for line in reader:
        rownames.append(line[0])
        mat.append(np.array(map(float, line[1: ])))
    return (np.array(mat), rownames, colnames)

def cosine(u, v):
    # Use scipy's method:
    return scipy.spatial.distance.cosine(u, v)
    # Or define it yourself:
    # return 1.0 - (np.dot(u, v) / (vector_length(u) * vector_length(v)))

def neighbors(word=None, mat=None, rownames=None, distfunc=cosine):
    if word not in rownames:
        raise ValueError('%s is not in this VSM' % word)
    w = mat[rownames.index(word)]
    dists = [(rownames[i], distfunc(w, mat[i])) for i in xrange(len(mat))]
    return sorted(dists, key=itemgetter(1), reverse=False)

def parseA_GLOVE(filename):
    num_lines = 0
    infile = open(filename, 'r')
    num_features = len(infile.readline().split()) - 1
    num_lines += 1
    for line in infile:
        assert len(line.split()) == num_features + 1
        num_lines += 1
    infile.close()
    mat = np.zeros((num_lines, num_features))
    vocab = []
    infile = open(filename, 'r')
    index = 0
    for line in infile:
        arr = line.split()
        vocab.append(arr[0])
        mat[index,:] = [float(num) for num in arr[1:]]
        index += 1
    infile.close()
    return mat, vocab

print 'Building GLOVE...'
#GLOVE_MAT, GLOVE_VOCAB = parseA_GLOVE(GLOVE_FILE)
GLOVE_MAT, GLOVE_VOCAB, _ = build('data/glove.6B.50d.txt', delimiter=' ', header=False, quoting=csv.QUOTE_NONE)

#print neighbors(word='niggas', mat=GLOVE_MAT, rownames=GLOVE_VOCAB)[: 5]

def glvvec(w):
    """Return the GloVe vector for w."""
    if GLOVE_CACHE != None:
        return GLOVE_CACHE[w]
    if w in GLOVE_VOCAB:
        i = GLOVE_VOCAB.index(w)
        return GLOVE_MAT[i]
    else:
        return np.zeros(GLVVEC_LENGTH)

def buildGloveCache(words):
    global GLOVE_CACHE
    print 'Building GLOVE cache...'
    temp = {}
    for w in words:
        temp[w] = glvvec(w)
    GLOVE_CACHE = temp

def glove_features_mean_unweighted(tweetRow, words):
    result = np.zeros(GLVVEC_LENGTH)
    count = 0.0
    for i, w in enumerate(words):
        if tweetRow[i] == 0.0: continue
        vec = glvvec(w)
        count += 1.0
        result += vec
    return result

def glove_features_mean_weighted(tweetRow, words):
    result = np.zeros(GLVVEC_LENGTH)
    count = 0.0
    for i, w in enumerate(words):
        vec = glvvec(w)
        count += tweetRow[i]
        result += tweetRow[i] * vec
    return result / count

# len(tweetRow) == len(words)
def glove_features(tweetRow, words):
    return glove_features_mean_unweighted(tweetRow, words)

def buildGloveTrainMat(train_file):
    wd = buildwd.buildWD(train_file, randomize=True)
    mat = wd[0]
    tweetIDs = wd[1]
    words = wd[2]
    labels = wd[3]
    buildGloveCache(words)
    mat = np.transpose(mat)
    print 'Building GLOVE train matrix...'
    trainMat = np.array([glove_features(mat[i,:], words) for i in range(len(tweetIDs))])
    return trainMat

def glove_knn(train_file, trainMat=None):
    if trainMat == None:
        trainMat = buildGloveTrainMat(train_file)

    wd = buildwd.buildWD(train_file, randomize=True)
    labels = wd[3]
    trainVals = buildwd.trainValsFromSubjects(labels)

    knn = neighbors.KNeighborsClassifier(n_neighbors=5)
    knn.fit(trainMat[0:(trainMat.shape[0]*0.7),:], trainVals[0:(trainMat.shape[0]*0.7)])
    return knn.score(trainMat[(trainMat.shape[0]*0.7):,:], trainVals[(trainMat.shape[0]*0.7):])

def glove_logreg(train_file, trainMat=None):
    logreg, trainMat, trainVals = get_glove_logreg(train_file, trainMat)
    return logreg.score(trainMat[(trainMat.shape[0]*0.7):,:], trainVals[(trainMat.shape[0]*0.7):])

def get_glove_logreg(train_file, trainMat=None):
    if trainMat == None:
        trainMat = buildGloveTrainMat(train_file)

    wd = buildwd.buildWD(train_file, randomize=True)
    labels = wd[3]
    trainVals = buildwd.trainValsFromSubjects(labels)

    logreg = linear_model.LogisticRegression()
    logreg.fit(trainMat[0:(trainMat.shape[0]*0.7),:], trainVals[0:(trainMat.shape[0]*0.7)])
    return logreg, trainMat, trainVals

if __name__ == "__main__":
    trainMat = buildGloveTrainMat(TRAIN_FILE)
    # score_knn = glove_knn(TRAIN_FILE, trainMat=trainMat)
    # print 'KNN: ', score_knn
    score_logreg = glove_logreg(TRAIN_FILE, trainMat=trainMat)
    print 'LogReg: ', score_logreg
