"""
Multi-task Deep Neural Network (DNN) Prediction Program

Inputs:
   - data: a path to data folder that stores the test datasets (.npz file) [Required]
           if the data folder only contains raw '_test.csv' files, then pre-process will automatically be done. 
   - model: a FullFilePath, in which a file named DeepNeuralNetParameters.npz stores the trained DNN model for the prediction [Required]
   - output destination: a FullFilePath to specify the output directory [Optional]
   - BatchSize: size of the batch of records to predict at once. [Optional]
   - label: 0 or 1, indcate whether the input data have true label or not. [Required]
   - dropoutProb: the probability of the nodes to be dropped randomly during predicting. [Optional]
   - rep: an integer, number of repetitions for random drop-out prediction. [Optional]
   - seed: random seed useful for dropout predictions. [Optional]
      
Output: in the specified output directory, 
   - a file named DeepNeuralNetPredict_log.txt that stores the log information of the testing process,
   - files named DeepNeuralNetPredict_datName.csv, where 'datName' is one of the QSAR task, that stores the predicted results of the inputted test data using the saved DNN model and related parameters. 
   - files named rsq_datName_test.csv, where 'datName' is one of the QSAR task, that stores the R-Squared between of prediction true activity of the test set and the prediction given by each of the output nodes.
   If use dropout predictions:
   - folders named DeepNeuralNetPredict_datName_dropout.csv, where 'datName' is one of the QSAR task, that stores the predicted results for each droppout prediction round.
 
Usage:
   - to get help:
     python DeepNeuralNetPredict_multi.py -h
   - to run the program: 
     python DeepNeuralNetPredict_multi.py [parameters] --data=FullpathToTestDataFile --model=FullPathToSavedDNNModel --result=FullPathToModelToPredictionOutput

Usage Example:
    (Detailed explanations in README)
    python DeepNeuralNetPredict.py --label=1 --data=data_sparse --model=models/multi_sparse
    python DeepNeuralNetPredict.py --label=1 --seed=0 --rep=10 --data=data_sparse --model=models/multi_sparse_2 --result=predictions/multi_sparse_2

------------- 
Prepared By  Junshui Ma  (Based on George Dahl's Kaggle Code)
06/12/2014

Last modified by Yuting Xu on Aug.07, 2017
"""

import numpy as num
import scipy.sparse as sp
import sys
import os
import argparse
import itertools

import gnumpy as gnp

from activationFunctions import *
import dnn
from DNNSharedFunc import *
from processData_sparse import *
import processData_dense
from DeepNeuralNetTrain import *

def collectPredictions(predStream):
    preds = list(predStream)
    return num.vstack(preds)

def getTestPreds(net, VariableParaDict, datasets, inpPreproFunct, useDropout = False, dense = False):
    """
    Return a list of the test set predictions for eath test data in 'datasets'
    """
    predsOrigSpace = []
    for i, ds in enumerate(datasets):
        testInpMbStream = (inpPreproFunct(x) for x in allMinibatches(512, ds.inps))
        if dense:
            avgMat = num.repeat(VariableParaDict['targMean'],ds.inps.shape[0],0)
            stdMat = num.repeat(VariableParaDict['targStd'],ds.inps.shape[0],0)
        else:
            avgMat = num.repeat(VariableParaDict['targMean'][num.newaxis,:],ds.inps.shape[0],0)
            stdMat = num.repeat(VariableParaDict['targStd'][num.newaxis,:],ds.inps.shape[0],0)
        testPreds = collectPredictions(net.predictions(testInpMbStream, True, useDropout)) * stdMat + avgMat
        predsOrigSpace.append(testPreds)
    return predsOrigSpace

def parseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", action="store", dest='dataPath', type=str, default = None, \
                        help = 'Path to the folder which contains test data sets, which are all pre-processed and stored in npz format.')
    parser.add_argument("--model", action="store", dest='model', type=str, default = '', \
                        help = 'FullFilePath of the model folder which contains a file DeepNeuralNetParameters.npz, which is in NPZ format and stores the trained Deep Neural Network. (default = (current directory))')
    parser.add_argument("--BatchSize", action="store", type=int, dest="BatchSize", default=256,\
                        help = "size of the batch of records to predict at once.(default=256)") 
    parser.add_argument("--label", action="store", type=int, dest='label', default=0,\
                        help = "Whether the data have true label or not. No=0, YES=1.")
    parser.add_argument("--dropout", "--dropouts", action="store", dest='dropoutStr', type=str, default = '-1',\
                        help = "layer-specific drop-out probability")
    parser.add_argument("--rep", action="store", type=int, dest='rep', default=0, \
                        help = "Number of repetitions for drop-out prediction. if == 0, means don't use dropout prediction")
    parser.add_argument("--seed", action="store", type=int, default=8, \
			help = "Seed the random number generator.")
    parser.add_argument("--dense", action = "store_true", default=False, \
                        help = "Whether use for dense QSAR tasks (default = False)")
    parser.add_argument("--result", type=str, dest='PredictResultPath', default = None, \
                        help = "Full filepath, in which the Predict result is saved")
    
    args = parser.parse_args()
    
    args.BatchSize = num.floor(args.BatchSize)
    assert(args.BatchSize>0)
    
    args.dropouts = [float(i) for i in args.dropoutStr.split("_")]
    assert(all(d <= 0.5  for d in args.dropouts))

    if args.PredictResultPath is None:
        args.PredictResultPath = args.model
    args.savePrefix = args.PredictResultPath

    args.modelPath = os.path.join(args.model, 'DeepNeuralNetParameters.npz')
    assert(os.path.exists(args.modelPath))
    
    assert(os.path.exists(args.dataPath))
    args.allTestDat = glob.glob(args.dataPath+"/*test.npz")
    args.allTestDat = sorted(args.allTestDat)
    
    return args

def main():
    # get inputs
    args = parseArgs()

    # set random seed
    if args.rep > 0:
        num.random.seed(args.seed)
    
    # create a folder to save prediction results if it doesn't exists
    if not os.path.exists(args.savePrefix):
        os.makedirs(args.savePrefix)
        
    # begin log
    logPath = os.path.join(args.savePrefix, "DeepNeuralNetPredict_log.txt")
    _logFile = open(logPath, 'w')
    log = Tee(sys.stdout, _logFile)

    print >>log, " ".join(sys.argv)
    print >>log, "Start time: %s " % (tstamp())

    # load trained model
    NNet, VariableParaDict = dnn.loadSavedNeuralNet(args.modelPath, args.dense)

    if len(args.dropouts)==1 :
        if (args.dropouts[0] > 0): # otherwise use same dropouts as in training 
            NNet.dropouts = [args.dropouts[0] for i in range(len(NNet.weights))]
        else:
            NNet.dropouts = VariableParaDict['dropouts']
            NNet.dropouts = NNet.dropouts.tolist()
    else:
        assert(len(NNet.weights)==len(args.dropouts))
        NNet.dropouts = args.dropouts

    # generate featTable from featNames in trained model
    featNames = VariableParaDict['featNames']
    featTable = dict((feat,j) for j,feat in enumerate(featNames))
    if args.dense:
        outputNames = VariableParaDict['outputNames']
        outputTable = dict((output,j) for j,output in enumerate(outputNames))
            
    # prepare test datasets
    if len(args.allTestDat)==0:
        # need to preprocess raw test datasets from .csv files
        if not args.dense:
            preprocess_test(args.dataPath,featTable)
        else:
            processData_dense.preprocess_test(args.dataPath,featTable,outputTable,args.label)
        args.allTestDat = glob.glob(args.dataPath+"/*test.npz")
        args.allTestDat = sorted(args.allTestDat)

    # QSAR task name of each test set 
    args.TestdatNames = []
    for p in args.allTestDat:
        if not args.dense:
            args.TestdatNames.append(os.path.basename(p).split("_")[0])
        else:
            args.TestdatNames.append(os.path.basename(p).split(".")[0])
        
    # load all test datasets from dataPath
    datasets = []
    for p in args.allTestDat:
        print >>log, "loading %s " % (p)
        datasets.append(Dataset(p, VariableParaDict['OutsSize'], None, -1))

    if VariableParaDict['transform'] == 'zscore':
        prior = 0.01
        inpsMean = VariableParaDict['trainInpsMean']
        inpsStddev = VariableParaDict['trainInpsStddev']
        for i, ds in enumerate(datasets):
            m = inpsMean[i]
            s = inpsStddev[i]
            ds.inps = (ds.inps.toarray().astype(dtype=num.float) - m[num.newaxis,:])/(prior + s[num.newaxis,:])
    
    preproInps = lambda xx: xx.toarray()
    if VariableParaDict['transform'] != None:
        if VariableParaDict['transform'] == 'zscore':
            #prior = 0.01
            print >>log, "Removing the saved train-data mean from each dimension and divide it by saved train-data standard deviation (plus %f)." % (prior)
            #inpsMean = VariableParaDict['trainInpsMean']
            #inpsStddev = VariableParaDict['trainInpsStddev']
            #preproInps = lambda xx: (xx.toarray() - inpsMean[num.newaxis,:])/(prior + inpsStddev[num.newaxis,:])
            preproInps = lambda xx: xx
        if VariableParaDict['transform'] == 'sqrt':
            print >>log, "Transforming inputs by taking the square root"
            preproInps = lambda xx: num.sqrt(xx.toarray())
        if VariableParaDict['transform'] == 'binarize':
            print >>log, "Transforming inputs by binarizing the input values"
            preproInps = lambda xx: (xx.toarray()>0.0)+0.0
        if VariableParaDict['transform'] == 'log':
            print >>log, "Transforming inputs by taking the logarithm after adding 1"
            preproInps = lambda xx: num.log(1.0 + xx.toarray())
        if VariableParaDict['transform'] == 'asinh':
            print >>log, "Transforming inputs by taking the inverse hyperbolic sine function"
            preproInps = lambda xx: num.arcsinh(xx.toarray())

    print >>log, "Start predict (no dropout)..."
    testPreds = getTestPreds(NNet, VariableParaDict, datasets, preproInps, False, args.dense)
    print >>log, "Finish prediction. Save results to folder %s " % (args.savePrefix)
    
    predictPath = []   
    for datName in args.TestdatNames:
        predictPath.append(os.path.join(args.savePrefix,'DeepNeuralNetPredict_%s.csv' % (datName)))

    for i,testPred in enumerate(testPreds):
        print >>log, "Save prediction results for %s test set."  % (args.TestdatNames[i])
        writePreds2(predictPath[i], datasets[i].molIds, testPred, header = VariableParaDict['datNames'],multicol=(VariableParaDict['OutsSize']>1))

    # If use dropout prediction
    if args.rep > 0:
        predictPath_dropout = []
        for datName in args.TestdatNames:
            folderToAdd = os.path.join(args.savePrefix,'DeepNeuralNetPredict_%s_dropout' % (datName))
            predictPath_dropout.append(folderToAdd)
            if not os.path.exists(folderToAdd):
                os.makedirs(folderToAdd)
        print >>log, "Start predict (with dropout)..."
        for r in range(args.rep): 
            print "Dropout prediction round: %d" % (r)
            testPreds_dropout = getTestPreds(NNet, VariableParaDict, datasets, preproInps, True, args.dense)
            for i,testPred in enumerate(testPreds_dropout):
                savePathForDropout = os.path.join(predictPath_dropout[i],'DeepNeuralNetPredict_%s_dropout_%d.csv' % (args.TestdatNames[i],r+1))
                writePreds2(savePathForDropout, datasets[i].molIds, testPred, header = VariableParaDict['datNames'],multicol=(VariableParaDict['OutsSize']>1))

    # If test set has true label
    if args.label != 0:
        for i,testPred in enumerate(testPreds):
            RSQs = num.ndarray(shape = (1, VariableParaDict['OutsSize']))
            for j in range(VariableParaDict['OutsSize']):
                if not args.dense:
                    RSQs[0,j] = rSq(testPred[:,j],datasets[i].origTargs)
                else:
                    RSQs[0,j] = rSq(testPred[:,j],datasets[i].origTargs[:,j])
            print >>log, "Save R-squared of prediction results for %s test set."  % (args.TestdatNames[i])
            writeMat(os.path.join(args.savePrefix, 'rsq_%s_test.csv' % (args.TestdatNames[i])), RSQs, header = VariableParaDict['datNames'])

    print >>log, "Finish time: %s " % (tstamp())
    _logFile.close()

if __name__ == "__main__":
    main()
