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
A group of functions for processing a group of SPARSE QSAR data sets used by DeepNeuralNet_QSAR Programs

Main function: 
    Pre-processing the raw data to facilitate later use.
    Given data folder path, convert all *.csv file to numpy *.npz file and save in the given save path . 
    Usage Illustration:
          python processData_sparse.py rawDataFolder processedDataFolder
    Usage Example:
          python processData_sparse.py data_sparse data_sparse_processed

Requirements for a single raw data file: 
    * In ".csv" format; 
    * The first column is MOLECULE names;
    * The second column is true activity (if known), with column name "Act"; 
	* The remaining columns are the input features (compound descriptor);
    * Name the file as "datName_training.csv" or "datName_test.csv", where "datName" is the QSAR task name;
    * The "datName" cannot contain "_". For example, you need to re-name the "RAT_F" task as "RAT-F". 
    
Requirements of organizing a group of data files:
    * It is ok to only have "datName_training.csv" but without the corresponding "datName_test.csv".
    * Put all the raw data sets under one folder. 

Last modified by Yuting Xu on Feb.8, 2017
"""

import gzip
import numpy as num
import glob
import scipy.sparse as sp
import cPickle as pk
import os
import sys

def smartOpen(path, mode='r'):
    if path.endswith(".gz"):
        return gzip.open(path, mode)
    return open(path, mode)

def featuresUsed(path):
    f = smartOpen(path, 'r')
    headerToks = map(lambda t:t.strip(), f.next().split(','))
    f.close()
    if headerToks[1] == "Act":
        return set(headerToks[2:])
    return set(headerToks[1:])

def buildGlobalFeatureTable(paths):
    allFeats = set()
    for path in paths:
        feats = featuresUsed(path)
        allFeats.update(feats)
    table = dict((feat,j) for j,feat in enumerate(allFeats))
    return table

def dsName(path):
    return str(os.path.basename(path).split("_")[0])

def loadRawDataset(path, featTable = None):
    if featTable is None:
        # for loading single data set to train single DNN
        featTable = buildGlobalFeatureTable([path])
    
    print "Loading %s" % (path)
    f = smartOpen(path, 'r')
    headerToks = map(lambda t:t.strip(), f.next().split(','))
    offset = 1 # since the first column is MOLECULE name
    if headerToks[1] == "Act":
        offset = 2
    mIds = []
    data, row, col = [], [], []
    targs = []
    r = 0
    for line in f:
        toks = map(lambda t:t.strip(), line.split(','))
        mIds.append(toks[0])
        if offset == 2:
            targs.append(float(toks[1]))
        for j in range(len(headerToks) - offset):
            x = float(toks[offset+j])
            key = headerToks[offset+j]
            if x != 0 and (key in featTable):
                c = featTable[headerToks[offset+j]]
                row.append(r)
                col.append(c)
                data.append(x)
        r += 1  
    f.close()
    
    data = num.array(data, dtype=num.int32)
    row = num.array(row, dtype=num.int)
    col = num.array(col, dtype=num.int)
    inps = sp.coo_matrix((data, (row, col)),shape=(r, len(featTable)))
    print "Converting inputs to csr ..."
    inps = inps.tocsr()

    if len(targs) > 0:
        targs = num.array(targs)
    else:
        targs = None
    mIds = num.array(mIds, dtype=num.object)
    featNames = sorted(featTable, key=featTable.get)
    
    return featNames, mIds, inps, targs

def packCSR(baseName, mat, d):
    """
    pack the csr matrix mat into dictionary d
    Used by function: CSVtoNPZ
    """
    d[baseName + "_data"] = mat.data
    d[baseName + "_indices"] = mat.indices
    d[baseName + "_indptr"] = mat.indptr
    d[baseName + "_shape"] = num.array(mat.shape)
    return d

def extractCSR(d, baseName):
    """
    Used by function: loadPackedData
    """
    data = d[baseName + "_data"]
    indices = d[baseName + "_indices"]
    indptr = d[baseName + "_indptr"]
    shape = tuple(d[baseName + "_shape"])
    return sp.csr_matrix((data, indices, indptr), shape=shape)

def loadPackedData(path):
    fd = open(path, 'rb')
    d = num.load(fd)
    molIds = d['molIds']
    featNames = d['featNames']
    inps = extractCSR(d, 'inps')
    targs = d['targs']  # possible to be "None".
    fd.close()
    return featNames, molIds, inps, targs

def removeAllZeroFeatures(X,featNames):
    assert(X.shape[1] == len(featNames))
    total = num.abs(num.array(X.sum(axis=0))).flatten()
    nonzero = num.arange(X.shape[1])[total>0]
    xx = X.tocsc()[:, nonzero]
    newfeatNames = map(lambda t:featNames[t], nonzero)
    return xx.tocsr(), newfeatNames

def prepareTestInpsColumn(testX, testFeatNames, trainFeatNames):
    assert(testX.shape[1] == len(testFeatNames))
    # build the feature table of training features.
    featTable = dict((feat,j) for j,feat in enumerate(trainFeatNames))
    # allocate an empty matrix with the right size. 
    testXX = sp.lil_matrix((testX.shape[0], len(trainFeatNames)), dtype=testX.dtype)
    for i in range(len(testFeatNames)):
        testFeatName = testFeatNames[i]
        if testFeatName in featTable.keys():
            testXX[:,featTable[testFeatName]]=testX[:,i]
    return testXX.tocsr()

def delete_rows(mat, indices):
    """
    Remove the rows denoted by "indices" from the numpy ndarray or CSR sparse matrix "mat"
    """
    if isinstance(mat, num.ndarray):
        mat = num.delete(mat,indices,0)
    elif isinstance(mat, sp.csr_matrix):
        mat = delete_rows_csr(mat, indices)
    else:
        raise ValueError("Cannot handle input data type!")
    return mat

def delete_rows_csr(mat, indices):
    """
    Remove the rows denoted by "indices" from the CSR sparse matrix "mat"
    """
    if not isinstance(mat, sp.csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    indices = list(indices)
    mask = num.ones(mat.shape[0],dtype=bool)
    mask[indices] = False
    return mat[mask]

def CSVtoNPZ(featTable, rawFilePath, saveFilePath):
    featNames, molIds, inps, targs = loadRawDataset(rawFilePath, featTable)
    toSave = {}
    toSave['featNames'] = featNames
    toSave['molIds'] = molIds
    packCSR('inps', inps, toSave)
    toSave['targs'] = targs
    print "Save processed data to: %s" % (saveFilePath)
    num.savez(saveFilePath, **toSave)

def preprocess_train(rawDataFolder):
    """
    pre-process raw data inside training function
    """
    basePath = rawDataFolder
    savePrefix = rawDataFolder
    allPaths = glob.glob(basePath+"/*.csv")
    trainPaths = glob.glob(basePath+"/*training.csv")
    trainPaths.sort()
    if len(trainPaths)==0:
        print >> sys.stderr, 'Cannot find training datasets!'
        return
    testPaths = glob.glob(basePath+"/*test.csv")
    testPaths.sort()

    if len(trainPaths) != len(testPaths): 
        print "---- Warning: No one-to-one matches between training set and test set. -----"

    featTable = buildGlobalFeatureTable(trainPaths)
    print "length of all feature table build from training sets = %d" % len(featTable)
    tblFile = open(os.path.join(savePrefix,"featTable.pk"),'w')
    pk.dump(featTable, tblFile)
    tblFile.close()

    for p in allPaths:
        print "------------"
        savePath = os.path.join(savePrefix,os.path.basename(p).split(".")[0]+".npz")
        CSVtoNPZ(featTable, p, savePath)
            
    print "------ Pre-process raw data finished. ------"

def preprocess_test(rawDataFolder,featTable):
    """
    pre-process raw data inside prediction function
    """
    basePath = rawDataFolder
    savePrefix = rawDataFolder

    testPaths = glob.glob(basePath+"/*test.csv")
    testPaths.sort()
    if len(testPaths)==0:
        print >> sys.stderr, 'Cannot find any raw datasets!'
        return

    print "length of all feature table load from model = %d" % len(featTable)

    for p in testPaths:
        print "------------"
        savePath = os.path.join(savePrefix,os.path.basename(p).split(".")[0]+".npz")
        CSVtoNPZ(featTable, p, savePath)
            
    print "------ Pre-process raw data finished. ------"

def main():
    basePath = sys.argv[1]
    
    savePrefix = sys.argv[2]
    if not os.path.exists(savePrefix):
        os.makedirs(savePrefix)
    
    allPaths = glob.glob(basePath+"/*.csv")
    trainPaths = glob.glob(basePath+"/*training.csv")
    testPaths = glob.glob(basePath+"/*test.csv")
    trainPaths.sort()
    testPaths.sort()
    
    featTable = buildGlobalFeatureTable(trainPaths)
    #featTable = buildGlobalFeatureTable(allPaths)
    print "length of all feature table = %d" % len(featTable)
    tblFile = open(os.path.join(savePrefix,"featTable.pk"),'w')
    pk.dump(featTable, tblFile)
    tblFile.close()

    if len(trainPaths) == len(testPaths):       
        for trainPath, testPath in zip(trainPaths, testPaths):
            ds = dsName(trainPath)
            print "======================"
            print "dataset name: %s" % ds
            assert(ds==dsName(testPath))

            for p in [trainPath, testPath]:
                savePath = os.path.join(savePrefix,os.path.basename(p).split(".")[0]+".npz")
                CSVtoNPZ(featTable, p, savePath)
    else:
        # in case that training sets and test sets are unmatched
        print "---- Warning: No one-to-one matches between training set and test set. -----"
        for p in allPaths:
            print "======================"
            savePath = os.path.join(savePrefix,os.path.basename(p).split(".")[0]+".npz")
            CSVtoNPZ(featTable, p, savePath)
            
    
if __name__ == "__main__":
    main()



