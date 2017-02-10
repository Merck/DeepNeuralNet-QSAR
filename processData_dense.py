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
A group of functions for processing a group of DENSE QSAR data sets used by DeepNeuralNet_QSAR Programs

Main function: 
    Pre-processing the raw data to facilitate later use.
    Given data folder path, convert all *.csv file to numpy *.npz file and save in the given save path . 
    Usage Illustration:
          python processData_dense.py rawDataFolder processedDataFolder numberOfOutputs
    Usage Example:
          python processData_dense.py data_dense data_dense_processed 3

Requirements for a single raw data file: 
    * In ".csv" format; 
    * The first column is MOLECULE names;
    * The second to #(1+numberOfOutputs) column is true activity (if known), with any column name; 
	* The remaining columns are the input features (compound descriptor);
    * Name the file as "training.csv" or "test.csv".
    
Requirements of organizing a group of data files:
    * It is ok to only have "training.csv" but without the corresponding "test.csv".
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
from processData_sparse import smartOpen, packCSR, extractCSR, delete_rows, delete_rows_csr
from DNNSharedFunc import writeMat

def featuresUsed(path,numberOfOutputs):
    f = smartOpen(path, 'r')
    headerToks = map(lambda t:t.strip(), f.next().split(','))
    f.close()
    offset = 1 + numberOfOutputs
    return set(headerToks[offset:])

def outputUsed(path,numberOfOutputs):
    f = smartOpen(path, 'r')
    headerToks = map(lambda t:t.strip(), f.next().split(','))
    f.close()
    offset = 1 + numberOfOutputs
    return set(headerToks[1:offset])

def buildGlobalTables(paths,numberOfOutputs):
    allFeats = set()
    alloutputs = set()
    for path in paths:
        feats = featuresUsed(path,numberOfOutputs)
        allFeats.update(feats)
        outs = outputUsed(path,numberOfOutputs)
        alloutputs.update(outs)
    featTable = dict((feat,j) for j,feat in enumerate(allFeats))
    outputTable = dict((out,j) for j,out in enumerate(alloutputs))
    return featTable, outputTable

def loadRawDataset(path, featTable, outputTable, numberOfOutputs):
    """
    numberOfOutputs = 0, for real test data which have no output columns
    """
    print "Loading %s" % (path)
    f = smartOpen(path, 'r')
    headerToks = map(lambda t:t.strip(), f.next().split(','))
    offset = 1 + numberOfOutputs # since the first column is MOLECULE name
    header_outputNames = headerToks[1:offset]

    mIds = []
    data, row, col = [], [], []
    targs = []
    r = 0
    for line in f:
        toks = map(lambda t:t.strip(), line.split(','))
        mIds.append(toks[0])
        if offset > 1:
            for j in range(1,offset):
                targs.append(float(toks[j]))
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
        targs = num.array(targs).reshape((r,numberOfOutputs))
    else:
        targs = None
    mIds = num.array(mIds, dtype=num.object)
    featNames = sorted(featTable, key=featTable.get)
    outputNames = sorted(outputTable, key=outputTable.get)

    if numberOfOutputs > 0:
        targs_ordered = num.zeros(shape=targs.shape)
        for i in range(numberOfOutputs):
            temp = outputTable.get(header_outputNames[i])
            targs_ordered[:,temp] = targs[:,i]
    else:
        targs_ordered = None
    
    return featNames, outputNames, mIds, inps, targs_ordered

def CSVtoNPZ(rawFilePath, saveFilePath, featTable, outputTable, numberOfOutputs):
    featNames, outputNames, molIds, inps, targs = loadRawDataset(rawFilePath, featTable, outputTable, numberOfOutputs)
    toSave = {}
    toSave['featNames'] = featNames
    toSave['outputNames'] = outputNames
    toSave['molIds'] = molIds
    packCSR('inps', inps, toSave)
    toSave['targs'] = targs
    print "Save processed data to: %s" % (saveFilePath)
    num.savez(saveFilePath, **toSave)

def loadPackedData(path):
    fd = open(path, 'rb')
    d = num.load(fd)
    molIds = d['molIds']
    featNames = d['featNames']
    outputNames = d['outputNames']
    inps = extractCSR(d, 'inps')
    targs = d['targs']  # possible to be "None".
    fd.close()
    return featNames, outputNames, molIds, inps, targs

def preprocess_train(rawDataFolder,numberOfOutputs):
    """
    pre-process DENSE raw data inside training function
    """
    basePath = rawDataFolder
    savePrefix = rawDataFolder

    allPaths = glob.glob(basePath+"/*.csv")
    trainPath = os.path.join(basePath,"training.csv")
    testPath = os.path.join(basePath,"test.csv")
    
    if not os.path.exists(trainPath):
        print >> sys.stderr, 'Cannot find training datasets!'
        return

    if not os.path.exists(testPath):
        print "---- Warning: No one-to-one matches between training set and test set. -----"

    featTable, outputTable = buildGlobalTables([trainPath],numberOfOutputs)
    print "length of all feature table = %d" % len(featTable)
    print "length of all output table = %d" % len(outputTable)

    tblFile = open(os.path.join(savePrefix,"featTable.pk"),'w')
    pk.dump(featTable, tblFile)
    tblFile.close()

    tblFile = open(os.path.join(savePrefix,"outputTable.pk"),'w')
    pk.dump(outputTable, tblFile)
    tblFile.close()

    savePath = os.path.join(savePrefix,"training.npz")
    CSVtoNPZ(trainPath, savePath, featTable, outputTable, numberOfOutputs)
    
    if os.path.exists(testPath):
        savePath = os.path.join(savePrefix,"test.npz")
        CSVtoNPZ(testPath, savePath, featTable, outputTable, numberOfOutputs)
            
    print "------ Pre-process raw data finished. ------"

def preprocess_test(rawDataFolder,featTable,outputTable,label = 0):
    """
    pre-process DENSE raw data inside prediction function
    """
    basePath = rawDataFolder
    savePrefix = rawDataFolder

    testPaths = glob.glob(basePath+"/*test.csv")
    testPaths.sort()
    if len(testPaths)==0:
        print >> sys.stderr, 'Cannot find any raw datasets!'
        return

    print "length of feature table load from model = %d" % len(featTable)
    if label != 0:
        print "length of output table load from model = %d" % len(outputTable)

    for p in testPaths:
        print "------------"
        savePath = os.path.join(savePrefix,os.path.basename(p).split(".")[0]+".npz")
        CSVtoNPZ(p, savePath, featTable, outputTable, (label!=0)*len(outputTable))
            
    print "------ Pre-process raw data finished. ------"


def main():
    basePath = sys.argv[1]
    savePrefix = sys.argv[2]
    numberOfOutputs = int(sys.argv[3])
    
    if not os.path.exists(savePrefix):
        os.makedirs(savePrefix)

    trainPath = os.path.join(basePath,"training.csv")
    testPath = os.path.join(basePath,"test.csv")
    
    featTable, outputTable = buildGlobalTables([trainPath],numberOfOutputs)
    print "length of all feature table = %d" % len(featTable)
    print "length of all output table = %d" % len(outputTable)

    tblFile = open(os.path.join(savePrefix,"featTable.pk"),'w')
    pk.dump(featTable, tblFile)
    tblFile.close()

    tblFile = open(os.path.join(savePrefix,"outputTable.pk"),'w')
    pk.dump(outputTable, tblFile)
    tblFile.close()

    savePath = os.path.join(savePrefix,"training.npz")
    CSVtoNPZ(trainPath, savePath, featTable, outputTable, numberOfOutputs)
    
    if os.path.exists(testPath):
        savePath = os.path.join(savePrefix,"test.npz")
        CSVtoNPZ(testPath, savePath, featTable, outputTable, numberOfOutputs)

if __name__ == "__main__":
    main()
