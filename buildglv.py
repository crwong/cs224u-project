import random
import buildwd
import numpy as np
import itertools

SUFFIX = 'small'
TRAIN_FILE = 'data/topics_%s/ALL_CLEAN_%s.txt' % (SUFFIX, SUFFIX)
WRITE_FILE = 'data/topics_%s/A_GLOVE_%s.txt' % (SUFFIX, SUFFIX)

def randmatrix(m, n, lower=-0.5, upper=0.5):
    """Creates an m x n matrix of random values in [lower, upper]"""
    return np.array([random.uniform(lower, upper) for i in range(m*n)]).reshape(m, n)

def glove(
        mat=None, rownames=None, 
        n=100, xmax=100, alpha=0.75, 
        iterations=100, learning_rate=0.05, 
        display_progress=False):
    """Basic GloVe. rownames is passed through unused for compatibility
    with other methods. n sets the dimensionality of the output vectors.
    xmax and alpha controls the weighting function (see the paper, eq. (9)).
    iterations and learning_rate control the SGD training.
    display_progress=True prints iterations and current error to stdout."""    
    m = mat.shape[0]
    W = randmatrix(m, n) # Word weights.
    C = randmatrix(m, n) # Context weights.
    B = randmatrix(2, m) # Word and context biases.
    indices = range(m)
    for iteration in range(iterations):
        error = 0.0        
        random.shuffle(indices)
        for i, j in itertools.product(indices, indices):
            if mat[i,j] > 0.0:     
                # Weighting function from eq. (9)
                weight = (mat[i,j] / xmax)**alpha if mat[i,j] < xmax else 1.0
                # Cost is J' based on eq. (8) in the paper:
                diff = np.dot(W[i], C[j]) + B[0,i] + B[1,j] - np.log(mat[i,j])                
                fdiff = diff * weight                
                # Gradients:
                wgrad = fdiff * C[j]
                cgrad = fdiff * W[i]
                wbgrad = fdiff
                wcgrad = fdiff
                # Updates:
                W[i] -= (learning_rate * wgrad) 
                C[j] -= (learning_rate * cgrad) 
                B[0,i] -= (learning_rate * wbgrad) 
                B[1,j] -= (learning_rate * wcgrad)                 
                # One-half squared error term:                              
                error += 0.5 * weight * (diff**2)
        if display_progress:
            print "iteration %s: error %s" % (iteration, error)
    # Return the sum of the word and context matrices, per the advice 
    # in section 4.2:
    return (W + C, rownames)

"""
Format is word then glove vector values with spaces used as delimiters
"""
def writeToFile(mat, rownames):
    f = open(WRITE_FILE, 'w')
    for i in range(len(rownames)):
        toWrite = rownames[i] + " "
        for j in range(mat.shape[1]):
            toWrite += str(mat[i,j])
            if j != mat.shape[1]-1:
                toWrite += " "
        toWrite += "\n"
        f.write(toWrite)
    f.close()

def buildGloveFile(mat, rownames):
    glv = glove(mat=mat, rownames=rownames)
    writeToFile(glv[0], glv[1])

if __name__ == '__main__':
    wd = buildwd.buildWD(TRAIN_FILE)
    mat = wd[0]
    rownames = wd[2]
    buildGloveFile(mat, rownames)
    
    
