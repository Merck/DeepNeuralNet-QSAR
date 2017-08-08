"""
Deep Neural Network (DNN) Training Program for DENSE QSAR dataset

Inputs: 
   - a list of parameters, and 
   - a path to data folder that stores the processed training datasets (.npz file) or raw datasets (.csv file) 
      
Output: in the specified output directory, 
   - a file named DeepNeuralNetParameters.npz that stores the trained NN model and related parameters, 
   - a file named DeepNeuralNetTrain_log.txt that stores the log information of the training process.
   - a file named MSEDuringTraining.csv that stores the Mean-Squared-Error of the training process
   If specify internal cross-validation proportation:
   - a file named MSEDuringTraining_CV.csv that stores the Mean-Squared-Error of all cross-validation sets during the training process
   - a file named rsqDuringTraining_CV.csv that stores the R-Squared of all cross-validation sets during the training process
   If given external test set:
   - a file named MSEDuringTraining_Test.csv that stores the Mean-Squared-Error of all test sets during the training process
   - a file named rsqDuringTraining_Test.csv that stores the R-Squared of all test sets during the training process
   If input data path only have raw datasets:
   - save the processed datasets (.npz files) within the original data folder.
   
Usage:
  - to get help
   python DeepNeuralNetTrain.py -h
  - to reun the program  
   python DeepNeuralNetTrain.py [parameters] --data=FullpathToDataFile FullPathToModelToOutput

Usage Example:
	(Detailed explanations in README)
    python DeepNeuralNetTrain_dense.py --CV=0.4 --test --keep=0_1 --watch=0 --hid=2000 --hid=1000 --dropouts=0_0.25_0.1 --epochs=10 --data=data_dense models/multi_dense_1
    python DeepNeuralNetTrain_dense.py --hid=2000 --hid=1000 --dropouts=0_0.25_0.1 --epochs=10 --data=data_dense models/multi_dense_2

-------------------------
Prepared By  Junshui Ma  (Based on George Dahl's Kaggle Code)
06/12/2014

Last modified by Yuting Xu on Feb.08, 2017
"""

import numpy as num
import scipy.sparse as sp
import sys
import os
import argparse
import itertools
import glob
import argparse

import gnumpy as gnp
import dnn
from DNNSharedFunc import *
from processData_dense import *
from DeepNeuralNetPredict import collectPredictions
from DeepNeuralNetTrain import zscoreByColumn

def sampleMB(mbsz, X, y = None):
    idx = num.random.randint(X.shape[0], size=(mbsz,))
    if y != None:
        return X[idx], y[idx,:]
    return X[idx]

def summarizePreds(net, datasets, inpPreproFunct, useDropout = False, datatype = 0):
    """
    Return mse and r-squared for CV or test data during training epochs.
    datatype = 0 for CV data; datatype = 1 for test data
    """
    MSEs_epoch = num.ndarray(shape=(1, datasets.targDims+1))
    RSQs_epoch = num.ndarray(shape=(1, datasets.targDims+1))

    if datatype == 0:
        InpMbStream = (inpPreproFunct(x) for x in allMinibatches(512, datasets.inpsCV))
        targs = datasets.targsCV
    else:
        InpMbStream = (inpPreproFunct(x) for x in allMinibatches(512, datasets.inpsTest))
        targs = datasets.targsTest
        
    Preds = collectPredictions(net.predictions(InpMbStream, True, useDropout))
    for i in range(datasets.targDims):
        MSEs_epoch[0,i] = calculateMSE(Preds[:,i],targs[:,i])
        RSQs_epoch[0,i] = rSq(Preds[:,i],targs[:,i])
        
    MSEs_epoch[0,-1] = num.mean(MSEs_epoch[0,:-1])
    RSQs_epoch[0,-1] = num.mean(RSQs_epoch[0,:-1])
    return MSEs_epoch, RSQs_epoch

def train(log, datasets, preproInps, preproTargs, args):
    net = buildDNN(args)
    prepro = lambda xx,yy: (preproInps(xx), preproTargs(yy))
    useDropout = any(x>0 for x in net.dropouts)

    if useDropout:
        print >>log, "Dropout probability in each layer:"
        print >>log, net.dropouts
        print >>log, "\n"

    mbStream = (prepro(*sampleMB(args.mbsz, datasets.inps, datasets.targs)) for unused in itertools.repeat(None))
            
    MSEs = list()
    MSEs_CV = num.ndarray(shape=(args.epochs, args.OutsSize+1))
    RSQs_CV = num.ndarray(shape=(args.epochs, args.OutsSize+1))
    MSEs_Test = num.ndarray(shape=(args.epochs, args.OutsSize+1))
    RSQs_Test = num.ndarray(shape=(args.epochs, args.OutsSize+1))
    print >>log, "Start training ......"
    print >>log, "\n"
    for ep, (err, unusd) in enumerate(net.fineTune(mbStream, args.epochs, args.mbPerEpoch, loss = None, progressBar = True, useDropout = useDropout)):
        mse = 2*err/args.OutsSize # since the error function the net optimizes is 0.5*(t-p)^2
        MSEs.append(mse)

        if args.CV > 0:
            MSEs_CV[ep,], RSQs_CV[ep,] = summarizePreds(net, datasets, preproInps, useDropout = False, datatype = 0)
        if args.test:
            MSEs_Test[ep,], RSQs_Test[ep,] = summarizePreds(net, datasets, preproInps, useDropout = False, datatype = 1)
        
        if args.CV <= 0 and (not args.test):
            print >>log, "Epoch %d:  MSE_training = %.3f" % (ep, mse)
        elif args.CV > 0 and (not args.test):
            print >>log, "Epoch %d:  MSE_training = %.3f,  MSE_CV = %.3f,  R-squared_CV = %.3f" % (ep, mse, MSEs_CV[ep,args.watch], RSQs_CV[ep,args.watch])
        elif args.CV <= 0 and (args.test):
            print >>log, "Epoch %d:  MSE_training = %.3f,  MSE_Test = %.3f,  R-squared_Test = %.3f" % (ep, mse, MSEs_Test[ep,args.watch], RSQs_Test[ep,args.watch])
        else:
            print >>log, "Epoch %d:  MSE_training = %.3f,  MSE_CV = %.3f,  R-squared_CV = %.3f,  MSE_Test = %.3f,  R-squared_Test = %.3f" % (ep, mse, MSEs_CV[ep,args.watch], RSQs_CV[ep,args.watch], MSEs_Test[ep,args.watch], RSQs_Test[ep,args.watch])
  
        if ep in args.anneal:
            print >>log, " Annealing learning rates. Dividing all learning rates by %f." % (args.annealingFactor)
            net.learnRates = [learnRate/args.annealingFactor for x in net.learnRates]

        log.flush()

    print >>log, ""
    return net, num.array(MSEs), MSEs_CV, RSQs_CV, MSEs_Test, RSQs_Test

def buildDNN(args):
    # initialize an DNN object instance
    if args.loadModel is None:
        args.layerSizes = [args.InpsSize] + args.hid + [args.OutsSize]
        net = dnn.DNN(args.layerSizes, dnn.Linear(), args.relu, None, None, args.targMean, args.targStd)
        net.dropouts = args.dropouts
    else:
        net, VariableParaDict = dnn.loadSavedNeuralNet(args.loadModel,True)
        print >>sys.stderr, "Loaded previous trained model from %s. " % (args.loadModel)
        if len(args.dropouts)==1 :
            if (args.dropouts[0] == -1):
                # use same dropouts as in loaded model
                net.dropouts = VariableParaDict['dropouts']
                net.dropouts = net.dropouts.tolist()
            else:
                net.dropouts = [args.dropouts[0] for i in range(len(net.weights))]
        else:
            assert(len(args.dropouts)==len(net.weights))
            net.dropouts = args.dropouts
        args.layerSizes = net.layerSizes
        datNames = VariableParaDict['datNames']
        for i in range(len(args.datNames)):
            assert(args.datNames[i] == datNames[i])

    # set training parameters
    net.learnRates = [args.learnRate for i in range(len(net.layerSizes))]
    net.momentum = args.momentum
    net.L2Costs = [args.weightCost for i in range(len(net.layerSizes))]
    net.nesterov = False
    net.nestCompare = False
    net.rmsLims = [None for i in range(len(net.layerSizes))]
    net.realValuedVis = (not (args.transform == 'binarize'))
    if net.realValuedVis and args.reducelearnRateVis:
        net.learnRates[0] = 0.005
    return net


class Dataset(object):
    def __init__(self, path, CV = -1, zscore = False):
        self.path = path

        featNames, outputNames, molIds, inps, targs = loadPackedData(path)
        self.molIds = molIds
        self.inps = inps
        self.origTargs = targs
        self.featNames = featNames
        self.outputNames = outputNames

        self.targDims = targs.shape[1]
        self.inpsDim = inps.shape[1]
        self.size = inps.shape[0]

        if (targs is not None) and bool(targs.any()):
            self.targMean = num.mean(targs,0).reshape((1,self.targDims))
            self.targStd = num.std(targs,0).reshape((1,self.targDims))
            avgMat = num.repeat(self.targMean,targs.shape[0],0)
            stdMat = num.repeat(self.targStd,targs.shape[0],0)
            self.targs = (self.origTargs - avgMat)/stdMat

        self.test = False
        self.CV = 0
        
        if CV > 0:
            self.CV = CV
            self.sizeCV = int(num.ceil(inps.shape[0] * CV))
            idCV = num.random.choice(inps.shape[0], size = self.sizeCV, replace = False)
            
            self.molIdsCV = self.molIds[idCV]
            self.inpsCV = self.inps[idCV]
            self.origTargsCV = self.origTargs[idCV]
            self.targsCV = self.targs[idCV]
       
            self.molIds = delete_rows(self.molIds,idCV)
            self.inps = delete_rows(self.inps,idCV)
            self.origTargs = delete_rows(self.origTargs,idCV)
            self.targs = delete_rows(self.targs,idCV)
     
            self.size = self.inps.shape[0]

        if zscore:
            self.inps = self.inps.toarray().astype(dtype=num.float)
            self.trainInpsMean = num.mean(self.inps, axis=0, dtype = num.float)
            self.trainInpsStddev = num.std(self.inps, axis=0, dtype = num.float)
            self.inps = zscoreByColumn(self.inps)
            if CV > 0:
                self.inpsCV = zscoreByColumn(self.inpsCV,self.trainInpsMean,self.trainInpsStddev)
        

    def perm(self):
        perm = num.random.permutation(self.inps.shape[0])
        self.molIds = self.molIds[perm]
        self.inps = self.inps[perm]
        self.origTargs = self.origTargs[perm]
        if (self.targs is not None) and bool(self.targs.any()):
            self.targs = self.targs[perm]

    def addTest(self, path, zscore = False):
        self.pathTest = path
        self.test = True
        featNamesTest, outputNamesTest, molIdsTest, inpsTest, targsTest = loadPackedData(path)
        self.molIdsTest = molIdsTest
        if zscore:
            inpsTest = zscoreByColumn(inpsTest,self.trainInpsMean,self.trainInpsStddev)
        self.inpsTest = inpsTest
        self.origTargsTest = targsTest
        self.outputNamesTest = outputNamesTest
        self.featNamesTest = featNamesTest

        self.sizeTest = inpsTest.shape[0]

        if (targsTest is not None) and bool(targsTest.any()):
            # standardize targs according to training data
            avgMat = num.repeat(self.targMean,targsTest.shape[0],0)
            stdMat = num.repeat(self.targStd,targsTest.shape[0],0)
            self.targsTest = (self.origTargsTest - avgMat)/stdMat
            
    def keepSelect(self,keep):
        self.origTargs = self.origTargs[:,keep]
        self.targs = self.targs[:,keep]
        self.outputNames = self.outputNames[keep]
        self.targDims = self.targs.shape[1]
        self.targMean = self.targMean[:,keep]
        self.targStd = self.targStd[:,keep]
        if self.CV > 0:
            self.origTargsCV = self.origTargsCV[:,keep]
            self.targsCV = self.targsCV[:,keep]
        if self.test:
            self.outputNamesTest = self.outputNamesTest[keep]
            self.origTargsTest = self.origTargsTest[:,keep]
            self.targsTest = self.targsTest[:,keep]
    

def parseArgs():
    #print " ".join(sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", action="store", type=int, default=8, \
			help = "Seed the random number generator.")
    parser.add_argument("--transform", default='log', choices = ['sqrt', 'log', 'asinh', 'binarize', 'zscore', None], \
                        help = 'Transform inputs. sqrt: sqrt(X); log: log(X+1); asinh: log(X+sqrt(1+X^2)); binarize: X=1 if X>0, otherise X=0. zscore: Standardized inputs by column. (default=log)')
    parser.add_argument("--hid", action="append", type=int, default=[],\
			help = "The number of nodes in each hidden layer.")
    parser.add_argument("--dropout", "--dropouts", action="store", dest='dropoutStr', type=str, default = '0', \
			help = "The drop-out probabilityies for input layer and hidden layers.")
    parser.add_argument("--relu", action="store_true", default=True, \
			help = "The activation function for hidden layer. True: use ReLU(); False: use Sigmoid().")
    parser.add_argument("--epochs", action="store", type=int, default=30, \
                        help = "Set the epoch size (default=30).")
    parser.add_argument("--learn-rate", action="store", dest='learnRate', type=float, default=0.05, \
                        help = "Set the learning rate (default=0.05).")
    parser.add_argument("--annealing-factor", dest='annealingFactor', action="store", type=float, default=2.0, \
			help = "Reduce the learning rate by this factor every few epochs.")
    parser.add_argument("--anneal", action="append", type=int, default=[], \
			help = "Anneal the learning rate at which epochs.")
    parser.add_argument("--weight-cost", action="store", dest='weightCost', type=float, default=0.0001, \
			help = "The strength of L2 regularization of weights.")
    parser.add_argument("--momentum", action="store", dest='momentum', type=float, default=0.9, \
			help = "The momentum parameter in gradient descent optimization algorithms.")
    parser.add_argument("--loadModel", action = "store", type=str, dest='loadModel', default=None, \
                        help = "Previous model path for loading and initializing the DNN, default=None.")					
    parser.add_argument("--CV", action = "store", type=float, dest='CV', default=-1, \
                        help = "Proportion of cross-validation data selected from each training data. Between 0 and 0.5, default = -1, don't use cross-validation feature.")
    parser.add_argument("--test", action = "store_true", default=False, \
                        help = "whether to import paired test data from same folder as training data.(default = False)")
    parser.add_argument("--reducelearnRateVis", action="store_true", default=False, \
			help = "whether reduce learn-rate at input layer for real value inputs.")
    parser.add_argument("--mbsz", action="store", type=int, dest="mbsz", default=20, \
                        help = "Set the minibatch size (default 20).")
    parser.add_argument("--numberOfOutputs", action="store", type=int, default=1, \
                        help = "Give the number of QSAR tasks in dense dataset. (default 1).")
    parser.add_argument("--keep", action="store", dest='keepStr', type=str, default='',\
                        help = "The output dimensions to keep use.")
    parser.add_argument("--watch", action="store", type=int, dest="watch", default= -1,\
                        help = "The output dimension to monitor during training. (default=-1, means all)") 
    parser.add_argument("--data", "--training-data", action="store", dest='dataPath', type=str, default = None, \
                        help = 'Full filename of the data file, which is in csv or csv.gz format and stores the training dataset.')
    parser.add_argument("TrainingResultPath", type=str, default = 'TrainedDNN', \
			help='Full filepath, in which the trained DNN model is saved')

    args = parser.parse_args()

    assert(all(0 < ep < args.epochs for ep in args.anneal))
    assert(args.annealingFactor >= 1)

    args.dropouts = [float(i) for i in args.dropoutStr.split("_")]
    assert(all(d <= 0.5  for d in args.dropouts))

    if(len(args.hid) != 0):
        assert(len(args.dropouts) == len(args.hid) + 1)
    else:
        assert(args.loadModel is not None) # use loaded model
        args.dropouts = [-1]
        
    assert(os.path.exists(args.dataPath))
    args.trainPath = os.path.join(args.dataPath,"training.npz")
    if not os.path.exists(args.trainPath):
        preprocess_train(args.dataPath, args.numberOfOutputs)
        args.trainPath = os.path.join(args.dataPath,"training.npz")

    featTablePath = os.path.join(args.dataPath,"featTable.pk")
    outputTablePath = os.path.join(args.dataPath,"outputTable.pk")
    if os.path.exists(featTablePath) and os.path.exists(outputTablePath):
        featTable = num.load(featTablePath)
        outputTable = num.load(outputTablePath)
        args.numberOfOutputs = len(outputTable)
    else:
        featTable, outputTable = buildGlobalTables([args.trainPath],args.numberOfOutputs)

    args.featNames = sorted(featTable, key=featTable.get)
    args.outputNames = sorted(outputTable, key=outputTable.get)

    if args.test:
        args.testPath = os.path.join(args.dataPath,"test.npz")
        if not os.path.exists(args.testPath):
            preprocess_test(args.dataPath,featTable,outputTable)
            args.testPath = os.path.join(args.dataPath,"test.npz")
        if not os.path.exists(args.testPath):
            args.test = False
            print >> sys.stderr, "Skip importing test set during training process!"
    
    args.savePrefix = args.TrainingResultPath

    if args.loadModel is not None:
        assert(os.path.exists(args.loadModel))
        if not args.loadModel.endswith(".npz"):
            # if a folder is given instead of a file
            args.loadModel = os.path.join(args.loadModel,'DeepNeuralNetParameters.npz')
            
    assert(args.CV < 1)

    if len(args.keepStr) > 0:
        args.keep = [int(i) for i in args.keepStr.split("_")]
    else:
        args.keep = None

    return args

def main():
    # get inputs
    args = parseArgs()
    # set random seed
    num.random.seed(args.seed)

    # create a folder to save training results if it doesn't exists
    if not os.path.exists(args.savePrefix):
        os.makedirs(args.savePrefix)
    # specify default file name for saving all neural net parameters
    saveModelPath = os.path.join(args.savePrefix,'DeepNeuralNetParameters.npz')
    # begin log
    logPath = os.path.join(args.savePrefix, "DeepNeuralNetTrain_log.txt")
    _logFile = open(logPath, 'w')
    log = Tee(sys.stdout, _logFile)
    
    print >>log, " ".join(sys.argv)
    print >>log, "Start time: %s " % (tstamp())
    
    # load training datasets
    print >>log, "loading %s " % (args.trainPath)
    datasets = Dataset(args.trainPath, args.CV, args.transform == 'zscore')
    args.InpsSize = datasets.inpsDim
    args.OutsSize = datasets.targDims
    args.trainingSize = datasets.size
    print >>log, "Training data: %d molecules, feature dimension: %d, outputs dimension: %d " % (args.trainingSize, args.InpsSize, args.OutsSize)
    
    # load paired test datasets
    if args.test:
        print >> log, "loading %s" % (args.testPath)
        datasets.addTest(args.testPath, args.transform == 'zscore')
        print >>log, "Test data: %d molecules" % (datasets.sizeTest)

    # only keep several output dimensions
    if args.keep is not None:
        assert(all(0 <= k <= args.OutsSize  for k in args.keep))
        print >>log, "Keep output dimensions in original data: "
        print >>log, args.keep
        args.OutsSize = len(args.keep)
        datasets.keepSelect(args.keep)
        print >>log, "Keep output dimensions names: "
        print >>log, datasets.outputNames
    args.datNames = datasets.outputNames
    
    # Calculate number of miniBatches per epoch (args.mbPerEpoch)
    args.mbPerEpoch = int(num.ceil(args.trainingSize/float(args.mbsz)))
    
    # save all targMean, targStd in args since they are important model parameters
    args.targMean = datasets.targMean
    args.targStd = datasets.targStd

    # if apply zscore transformation, save all trainInpsMean, trainInpsStddev in args since they are important model parameters
    if args.transform == 'zscore':
        args.trainInpsMean = []
        args.trainInpsStddev = []
        args.trainInpsMean.append(datasets.trainInpsMean)
        args.trainInpsStddev.append(datasets.trainInpsStddev)
    
    # transform input features
    if args.transform != None:
        if args.transform == 'sqrt':
            print >>log, "Transforming inputs by taking the square root."
            preproInps = lambda xx: num.sqrt(xx.toarray())
        if args.transform == 'binarize':
            print >>log, "Transforming inputs by binarizing the input values."
            preproInps = lambda xx: (xx.toarray()>0.0)+0.0
        if args.transform == 'log':
            print >>log, "Transforming inputs by taking the logarithm after adding 1."
            preproInps = lambda xx: num.log(xx.toarray() + 1.0)
        if args.transform == 'asinh':
            print >>log, "Transforming inputs by taking the inverse hyperbolic sine function."
            preproInps = lambda xx: num.arcsinh(xx.toarray())
        if args.transform == 'zscore':
            print >>log, "Transforming inputs by standardizing each column (feature)."
            preproInps = lambda xx: xx
    else:
        print >>log, "No transformation performed on inputs."
        preproInps = lambda xx: xx.toarray()

    # prepare target output activity
    preproTargs = lambda yy: yy
                        
    # print some training details
    print >>log, "%d records per miniBatch, %d miniBatches per epoch, and %d epochs." % (args.mbsz, args.mbPerEpoch, args.epochs)
    print >>log, "Number of nodes in hidden Layers:" 
    if args.loadModel is None:
        print >>log, args.hid
    else:
        print >>log, "(same as loaded model)"
    if args.CV > 0:
        print >>log, "Percentage of leave-out cross-validation data in original training data: %.1f%%" % (100*args.CV)
    if args.watch != -1:
        print >>log, "Monitor output dimension: %d " % args.watch
    print >>log, "\n"
    
    # Core training process begin!
    net, MSEs, MSEs_CV, RSQs_CV, MSEs_Test, RSQs_Test = train(log, datasets, preproInps, preproTargs, args)

    # Finished and save
    print >>log, "Saving the trained Neural Net to %s" % saveModelPath
    
    NNParametersDict = net.NeuralNetParametersDict()
    NNParametersDict = saveArgs(NNParametersDict,args)
    num.savez(saveModelPath, **NNParametersDict);
    
    mseFile = os.path.join(args.savePrefix, "MSEDuringTraining.csv")
    writeAColumn(mseFile, MSEs, header = "MSE")

    if args.CV > 0:
        mseCVFile = os.path.join(args.savePrefix, "MSEDuringTraining_CV.csv")
        writePredsSummary(mseCVFile, MSEs_CV, datasets.outputNames)
        rsqCVFile = os.path.join(args.savePrefix, "rsqDuringTraining_CV.csv")
        writePredsSummary(rsqCVFile, RSQs_CV, datasets.outputNames)
    if args.test:
        mseTestFile = os.path.join(args.savePrefix, "MSEDuringTraining_Test.csv")
        writePredsSummary(mseTestFile, MSEs_Test, datasets.outputNames)
        rsqTestFile = os.path.join(args.savePrefix, "rsqDuringTraining_Test.csv")
        writePredsSummary(rsqTestFile, RSQs_Test, datasets.outputNames)
        
    print >>log, "Finish time: %s " % (tstamp())
    _logFile.close()


if __name__ == "__main__":
    main()
