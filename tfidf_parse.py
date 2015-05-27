import buildwd
import numpy as np

def tfidf(mat=None, rownames=None):
    """TF-IDF on mat. rownames is unused; it's an argument only 
    for consistency with other methods used here"""
    colsums = np.sum(mat, axis=0)
    doccount = mat.shape[1]
    w = np.array([_tfidf_row_func(row, colsums, doccount) for row in mat])
    return (w, rownames)

def _tfidf_row_func(row, colsums, doccount):
    df = float(len([x for x in row if x > 0]))
    idf = 0.0
    # This ensures a defined IDF value >= 0.0:
    if df > 0.0 and df != doccount:
        idf = np.log(doccount / df)
    tfs = row/colsums
    return tfs * idf

"""
File should have a tweet on each line. Each line should contain id, subject, tweet.
"""
def tfidf_parse(file_name):
    wd = buildwd.buildWD(file_name)
    print "Starting tfidf"
    return tfidf(wd[0])

idf = tfidf_parse("training.txt")
            

