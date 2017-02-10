"""
    Copyright (c) 2011,2012,2016,2017 Merck Sharp & Dohme Corp. a subsidiary of Merck & Co., Inc., Kenilworth, NJ, USA.

    This file is part of the Deep Neural Network QSAR program.

    Deep Neural Network QSAR is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""

"""
A group of functions used by Deep Neural Net Programs.
Prepared By  Junshui Ma  (Based on George Dahl's Kaggle Code).
Last modified by Yuting Xu on Feb. 08, 2017.
"""

import gzip
import numpy as num
import gnumpy as gnp
import scipy.sparse as sp
import time
import sys

def collectPredictions(predStream):
    """
    Use by both train and predict functions.
    """
    preds = list(predStream)
    return num.vstack(preds)

def saveArgs(NNParametersDict,args):
    argsDict = vars(args)
    for key,value in argsDict.items():
        if key not in NNParametersDict:
            NNParametersDict[key] = value
    return NNParametersDict

def tstamp():
    t = time.localtime()
    return "%d/%d/%d %d:%.2d.%d" % ((t[1],t[2],t[0])+t[3:6])

def nonZeroDataByColumn(X):
    """
    Use by DeepNeuralNetTrain and DeepNeuralNetTrain_multi, if args.transform == 'standardize'
    """
    cscData = X.tocsc()
    columns = num.empty((cscData.shape[1],), dtype=num.object)
    for j in range(len(columns)):
        columns[j] = cscData.data[cscData.indptr[j]:cscData.indptr[j+1]].copy()
    return columns

def denseChunkMap(funct, mbsz, X):
    """
    X should be a CSR format scipy.sparse matrix. This function
    iterates over chunks of mbsz rows of X, makes them dense, applies
    funct to them, and makes them sparse again.
    """
    numChunks = int(num.ceil(X.shape[0]/float(mbsz)))
    for m in range(numChunks):
        yield sp.csr_matrix(funct(X[m*mbsz:min((m+1)*mbsz, X.shape[0])].toarray()))

def allMinibatches(mbsz, X, y = None):
    # used in predict!
    numBatches = int(num.ceil(X.shape[0]/float(mbsz)))
    for i in range(numBatches):
        if y != None:
            yield X[mbsz*i:min(mbsz*(i+1), X.shape[0])], y[mbsz*i:mbsz*(i+1)][:, num.newaxis]
        else:
            yield X[mbsz*i:min(mbsz*(i+1), X.shape[0])]

def rSq(preds, targs):
    assert(preds.shape == targs.shape)
    avp = preds.mean()
    avt = targs.mean()
    numer = num.sum( (targs - avt)*(preds - avp)  )
    numer *= numer
    denom = num.sum( (targs - avt)*(targs - avt) ) * num.sum((preds - avp)*(preds - avp))
    return numer/denom

def calculateMSE(preds,targs):
    assert(preds.shape == targs.shape)
    return num.mean( (preds-targs)*(preds-targs) )

def writePreds(outPath, molIds, preds, header,multicol = False):
    assert(molIds.shape[0] == preds.shape[0])
    f = open(outPath, 'w')
    if multicol:
        if header:
            print >>f, '"MOLECULE",',
            for i in range(preds.shape[1]-1):
                print >>f, 'Prediction_%d,' % (i+1),
            print >>f, 'Prediction_%d' % (preds.shape[1])
        for i in range(len(preds)):
            print >>f, '"%s",' % (molIds[i]),
            for j in range(preds.shape[1]-1):
                print >>f, '%f,' % (preds[i,j]),
            print >>f, '%f' % (preds[i,(preds.shape[1]-1)])
    else:
        if header:
            print >>f, '"MOLECULE","Prediction"'
        for i in range(len(preds)):
            print >>f, '"%s",%f' % (molIds[i], preds[i])
    f.close()

def writePreds2(outPath, molIds, preds, header = None, multicol = False):
    assert(molIds.shape[0] == preds.shape[0])
    f = open(outPath, 'w')
    if multicol:
        if header is not None:
            print >>f, '"MOLECULE",',
            for i in range(preds.shape[1]-1):
                #print >>f, 'Prediction_%s,' % (header[i]),
                print >>f, '%s,' % (header[i]),
            #print >>f, 'Prediction_%s' % (header[i+1])
            print >>f, '%s' % (header[i+1])
        for i in range(len(preds)):
            print >>f, '"%s",' % (molIds[i]),
            for j in range(preds.shape[1]-1):
                print >>f, '%f,' % (preds[i,j]),
            print >>f, '%f' % (preds[i,(preds.shape[1]-1)])
    else:
        if header is not None:
            #print >>f, '"MOLECULE","Prediction_%s"' % (header[0])
            print >>f, '"MOLECULE","%s"' % (header[0])
        for i in range(len(preds)):
            print >>f, '"%s",%f' % (molIds[i], preds[i])
    f.close()

def writePredsSummary(outPath, mat, datNames):
    """
    write MSEs and R-squared table for multi-task training
    """
    assert(mat.shape[1] == len(datNames) + 1)
    f = open(outPath, 'w')
    # print header
    print >>f, '"Iter",',
    for i in range(len(datNames)):
        print >>f, '%s,' % datNames[i],
    print >>f, 'Average'
    # print mat
    for i in range(mat.shape[0]):
        print >>f, '"%d",' % (i),
        for j in range(mat.shape[1]-1):
            print >>f, '%f,' % (mat[i,j]),
        print >>f, '%f' % (mat[i,mat.shape[1]-1])
    f.close()

def writeMat(outPath, mat, header = None):
    """
    write a matrix for general use. no column names or row names.
    """
    f = open(outPath, 'w')
    # print header
    if header is not None:
        assert(mat.shape[1] == len(header))
        for j in range(len(header)-1):
            print >>f, '%s,' % header[j],
        print >>f, '%s' % header[len(header)-1]
    # print mat
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]-1):
            print >>f, '%.4f,' % (mat[i,j]),
        print >>f, '%.4f' % (mat[i,mat.shape[1]-1])
    f.close()

def writeAColumn(outPath, NumArray, header = None):
    f = open(outPath, 'w')
    if header is not None:
        #print >>f, '"Epoch","MSE"'
        print >>f, '"Epoch", %s' % (header)
    for i in range(len(NumArray)):
        print >>f, '%d,%f' % (i, NumArray[i])
    f.close()

def writeRsq(outPath, rr):
    f = open(outPath, 'w')
    print >>f, '"R-squared"'
    if isinstance(rr, float):
        print >>f, '%f' % (rr)
    else:
        for i in range(len(rr)):
            print >>f, '%f' % (rr[i])
    f.close()

def printMaxGrad(net,log):
    for i in range(len(net.weights)):
        print >>log, "  Maximum Weight and Bias Gradient in layer %d: %f and %f" % (i,num.array(gnp.max(abs(net.WGrads[i]))),num.array(gnp.max(abs(net.biasGrads[i]))))
    print >>log, "==========="

class Tee(object):
    def __init__(self, *args):
        self.files = [f for f in args]
    def write(self, s):
        for f in self.files:
            f.write(s)
    def flush(self):
        for f in self.files:
            f.flush()
