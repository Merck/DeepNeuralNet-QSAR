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
Multi-task Deep Neural Network (DNN) Training Program - for sparse dataset(s)

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
  - to run the program  
   python DeepNeuralNetTrain.py [parameters] --data=FullpathToDataFile FullPathToModelToOutput

Usage Examples:
	(Detailed explanations in README)
	python DeepNeuralNetTrain.py --seed=0 --hid=2000 --hid=1000 --dropouts=0_0.25_0.1 --epochs=10 --data=data_sparse_single models/METAB_single
	python DeepNeuralNetTrain.py --seed=0 --CV=0.4 --test --hid=2000 --hid=1000 --dropouts=0_0.25_0.1 --epochs=10 --data=data_sparse_single models/METAB_single
	python DeepNeuralNetTrain.py --seed=0 --hid=2000 --hid=1000 --dropouts=0_0.25_0.1 --epochs=10 --data=data_sparse models/multi_sparse
	python DeepNeuralNetTrain.py --seed=0 --hid=2000 --hid=1000 --dropouts=0_0.25_0.1 --epochs=10 --data=data_sparse --loadModel=models/multi_sparse models/multi_sparse_continue
	python DeepNeuralNetTrain.py --seed=0 --CV=0.4 --test --mbsz=30 --keep=METAB --keep=OX1 --watch=OX1 --hid=2000 --hid=1000 --dropouts=0_0.25_0.1 --epochs=10 --data=data_sparse models/multi_sparse_2

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
from processData_sparse import *
from DeepNeuralNetPredict import collectPredictions

def summarizePreds(net, datasets, inpPreproFunct, useDropout = False, datatype = 0):
    """
    Return mse and r-squared for CV or test data during training epochs.
    datatype = 0 for CV data; datatype = 1 for test data
    """
    MSEs_epoch = num.ndarray(shape=(1, len(datasets)+1))
    RSQs_epoch = num.ndarray(shape=(1, len(datasets)+1))
    for i, ds in enumerate(datasets):
        if datatype == 0:
            InpMbStream = (inpPreproFunct(x) for x in allMinibatches(512, ds.inpsCV))
            targs = ds.targsCV
        else:
            InpMbStream = (inpPreproFunct(x) for x in allMinibatches(512, ds.inpsTest))
            targs = ds.targsTest
        Preds = collectPredictions(net.predictions(InpMbStream, True, useDropout))[:,i]
        MSEs_epoch[0,i] = calculateMSE(Preds,targs)
        RSQs_epoch[0,i] = rSq(Preds,targs)
    MSEs_epoch[0,-1] = num.mean(MSEs_epoch[0,:-1])
    RSQs_epoch[0,-1] = num.mean(RSQs_epoch[0,:-1])
    return MSEs_epoch, RSQs_epoch

def train(log, datasets, preproInps, preproTargs, args):
    net = buildDNN(args)
    prepro = lambda xx,yy,mm: (preproInps(xx), preproTargs(yy), mm)
    useDropout = any(x>0 for x in net.dropouts)

    if useDropout:
        print >>log, "Dropout probability in each layer:"
        print >>log, net.dropouts
        print >>log, "\n"

    if args.useAll:
        mbStream = (prepro(*allMB_multi(args.mbsz, datasets, mbNumber)) for mbNumber in itertools.cycle(range(args.mbPerEpoch)))
    else:
        mbStream = (prepro(*sampleMBFromAll(args.mbsz, datasets)) for unused in itertools.repeat(None))
            
    MSEs = list()
    MSEs_CV = num.ndarray(shape=(args.epochs, args.OutsSize+1))
    RSQs_CV = num.ndarray(shape=(args.epochs, args.OutsSize+1))
    MSEs_Test = num.ndarray(shape=(args.epochs, args.OutsSize+1))
    RSQs_Test = num.ndarray(shape=(args.epochs, args.OutsSize+1))
    print >>log, "Start training ......"
    print >>log, "\n"
    for ep, (err, unusd) in enumerate(net.fineTune(mbStream, args.epochs, args.mbPerEpoch, loss = None, progressBar = True, useDropout = useDropout)):
        mse = 2*err # since the error function the net optimizes is 0.5*(t-p)^2
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
        net = dnn.DNN(args.layerSizes, dnn.LinearMasked(), args.relu, None, None, args.targMean, args.targStd)
        net.dropouts = args.dropouts
    else:
        net, VariableParaDict = dnn.loadSavedNeuralNet(args.loadModel)
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
    def __init__(self, path, targDims, dsId, CV = -1):
        self.datName = os.path.basename(path).split("_")[0]
        self.path = path
        self.targDims = targDims
        self.dsId = dsId
        
        featNames, molIds, inps, targs = loadPackedData(path)
        self.molIds = molIds
        self.inps = inps
        self.origTargs = targs
        self.featNames = featNames

        self.inpsDim = inps.shape[1]
        self.size = inps.shape[0]

        if (targs is not None) and bool(targs.any()):
            self.targMean = targs.mean()
            self.targStd = targs.std()
            self.targs = (self.origTargs - self.targMean)/self.targStd
            if dsId is not None:
                self.targsFull = num.zeros((self.targs.shape[0], targDims), dtype=num.float32)
                self.targsFull[:,dsId] = self.targs

        if CV > 0:
            self.sizeCV = int(num.ceil(inps.shape[0] * CV))
            idCV = num.random.choice(inps.shape[0], size = self.sizeCV, replace = False)
            
            self.molIdsCV = self.molIds[idCV]
            self.inpsCV = self.inps[idCV]
            self.origTargsCV = self.origTargs[idCV]
            self.targsCV = self.targs[idCV]
            self.targsFullCV = self.targsFull[idCV]
            
            self.molIds = delete_rows(self.molIds,idCV)
            self.inps = delete_rows(self.inps,idCV)
            self.origTargs = delete_rows(self.origTargs,idCV)
            self.targs = delete_rows(self.targs,idCV)
            if dsId is not None:
                self.targsFull = delete_rows(self.targsFull,idCV)
            
            self.size = self.inps.shape[0]
        
    def perm(self):
        perm = num.random.permutation(self.inps.shape[0])
        self.molIds = self.molIds[perm]
        self.inps = self.inps[perm]
        self.origTargs = self.origTargs[perm]
        if (self.targs is not None) and bool(self.targs.any()):
            self.targs = self.targs[perm]
            self.targsFull = self.targsFull[perm]

    def addTest(self, path):
        assert(self.datName == os.path.basename(path).split("_")[0]) # check the name of test set is the same as training set
        self.pathTest = path
        featNamesTest, molIdsTest, inpsTest, targsTest = loadPackedData(path)
        self.molIdsTest = molIdsTest
        self.inpsTest = inpsTest
        self.origTargsTest = targsTest
        self.featNamesTest = featNamesTest

        if (targsTest is not None) and bool(targsTest.any()):
            self.targsTest = (self.origTargsTest - self.targMean)/self.targStd # standardize targs according to training data

def sampleMBFromAll(casesPerTask, datasets):
    inpsList = []
    targs = num.zeros((sum(casesPerTask), len(datasets)), dtype=num.float32)
    targsMask = num.zeros((sum(casesPerTask), len(datasets)), dtype=num.float32)
    for i in range(len(datasets)):
        idx = num.random.randint(datasets[i].inps.shape[0], size=(casesPerTask[i],))
        inpsList.append(datasets[i].inps[idx])
        targs[sum(casesPerTask[:i]):sum(casesPerTask[:(i+1)])] = datasets[i].targsFull[idx]
        targsMask[sum(casesPerTask[:i]):sum(casesPerTask[:(i+1)]), i] = 1
    inps = sp.vstack(inpsList)
    return inps, targs, targsMask

def allMB_multi(casesPerTask,datasets,mbNumber):
    if mbNumber == 0:
        # to begin a new epoch, permute each dataset first, then sequencially use training data in new order
        for i in range(len(datasets)):
            datasets[i].perm()

    inpsList = []
    targs = num.zeros((sum(casesPerTask), len(datasets)), dtype=num.float32)
    targsMask = num.zeros((sum(casesPerTask), len(datasets)), dtype=num.float32)
    for i in range(len(datasets)):
        # in case that we need to use certain datasets multiple times in one epoch
        idx = [ xx % (datasets[i].inps.shape[0]) for xx in range(casesPerTask[i]*(mbNumber-1), casesPerTask[i]*mbNumber)]
        inpsList.append(datasets[i].inps[idx])
        targs[sum(casesPerTask[:i]):sum(casesPerTask[:(i+1)])] = datasets[i].targsFull[idx]
        targsMask[sum(casesPerTask[:i]):sum(casesPerTask[:(i+1)]), i] = 1
    inps = sp.vstack(inpsList)
    return inps, targs, targsMask    
    

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", action="store", type=int, default=8, \
			help = "Seed the random number generator.")
    parser.add_argument("--transform", default='log', choices = ['sqrt', 'log', 'asinh', 'binarize', None], \
                        help = 'Transform inputs. sqrt: sqrt(X); log: log(X+1); asinh: log(X+sqrt(1+X^2)); binarize: X=1 if X>0, otherise X=0. (default=log)')
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
    parser.add_argument("--useAll", action = "store_true", default=False, \
                        help = "Whether use all training data in each epoch. True: use allMB_multi().")
    parser.add_argument("--CV", action = "store", type=float, dest='CV', default=-1, \
                        help = "Proportion of cross-validation data selected from each training data. Between 0 and 0.5, default = -1, don't use cross-validation feature.")
    parser.add_argument("--test", action = "store_true", default=False, \
                        help = "whether to import paired test data from same folder as training data.(default = False)")
    parser.add_argument("--reducelearnRateVis", action="store_true", default=False, \
			help = "whether reduce learn-rate at input layer for real value inputs.")
    parser.add_argument("--mbsz", action="store", type=str, dest="mbszStr", default='20', \
                        help = "Set the minibatch size, defined as the number of cases PER DATASET in a minibatch. if == -1, make the numer of cases per dataset proportational to the dataset size.")   
    parser.add_argument("--keep", action="append", type=str, default=[],\
                        help = "The datasets to keep use. (for multi-task only)")
    parser.add_argument("--watch", action="store", type=str, dest="watchStr", default=None,\
                        help = "The dataset to monitor during training. (for multi-task only, default=None)") 
    parser.add_argument("--data", "--training-data", action="store", dest='dataPath', type=str, default=None, \
                        help = "Full filename of the data file, which is in csv or csv.gz format and stores the training dataset.")
    parser.add_argument("TrainingResultPath", type=str, default = 'TrainedDNN', \
			help='Full filepath, in which the trained DNN model is saved')

    args = parser.parse_args()

    args.mbsz = [int(i) for i in args.mbszStr.split("_")]

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

    args.allTrainingDat = glob.glob(args.dataPath+"/*training.npz")
    if len(args.allTrainingDat)==0:
        preprocess_train(args.dataPath)
        args.allTrainingDat = glob.glob(args.dataPath+"/*training.npz")
    args.allTrainingDat = sorted(args.allTrainingDat)

    args.datNames = []
    for p in args.allTrainingDat:
        args.datNames.append(os.path.basename(p).split("_")[0])

    if args.test:
        args.allTestDat = []
        for onedatname in args.datNames:
            args.allTestDat.append(args.dataPath+"/"+onedatname+"_test.npz")

    featTablePath = os.path.join(args.dataPath,"featTable.pk")
    if os.path.exists(featTablePath):
        featTable = num.load(featTablePath)
    else:
        featTable = buildGlobalFeatureTable(args.allTrainingDat)

    args.featNames = sorted(featTable, key=featTable.get)

    args.savePrefix = args.TrainingResultPath

    if args.loadModel is not None:
        assert(os.path.exists(args.loadModel))
        if not args.loadModel.endswith(".npz"):
            # if a folder is given instead of a file
            args.loadModel = os.path.join(args.loadModel,'DeepNeuralNetParameters.npz')
            
    assert(args.CV < 1)

    if len(args.keep) > 0:
        deleteindices = []
        for i,datname in enumerate(args.datNames):
            if datname not in args.keep:
                deleteindices.append(i)
        for i in sorted(deleteindices, reverse = True):
            del args.datNames[i]
            del args.allTrainingDat[i]
            if args.test:
                del args.allTestDat[i]

    if args.watchStr is None:
        args.watch = -1
    else:
        for i,datname in enumerate(args.datNames):
            if datname == args.watchStr:
                args.watch = i

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
    
    # load all training datasets
    datasets = []
    dsId = 0
    args.datasets_sizes = []
    for p in args.allTrainingDat:
        print >>log, "loading %s " % (p)
        datasets.append(Dataset(p, len(args.allTrainingDat), dsId, args.CV))
        args.datasets_sizes.append(datasets[-1].size)
        dsId += 1
    args.InpsSize = datasets[0].inpsDim
    args.OutsSize = len(datasets)
    
    # load paired test datasets
    dsId = 0
    if args.test:
        for p in args.allTestDat:
            if not os.path.exists(p):
                print >> log, "Cannot find %s. Use training set as test set. " % (p)
                p = os.path.join(args.dataPath,os.path.basename(p).split("_")[0]+"_training.npz")               
            print >> log, "loading %s" % (p)
            datasets[dsId].addTest(p)
            dsId += 1

    # Calculate miniBatch sizes (args.mbsz) and number of miniBatches per epoch (args.mbPerEpoch)
    if (args.mbsz[0] < 0):
        # make the mbsz per dataset proportional to each dataset size, but at least 5 per dataset
        args.totalmbsz = sum(args.datasets_sizes) * 5 / min(args.datasets_sizes)
        args.mbsz = [int(args.totalmbsz*size_each/sum(args.datasets_sizes)) for size_each in args.datasets_sizes]
    else:
        if (len(args.mbsz)==1):
            args.mbsz = [args.mbsz[0] for i in range(len(datasets))]
        else:
            assert(len(args.mbsz) == len(datasets))
            
    args.totalmbsz = sum(args.mbsz)           
    args.mbPerEpoch = max([int(num.ceil(args.datasets_sizes[i]/float(args.mbsz[i]))) for i in range(len(datasets))]) # so that nearly all data can be use once each epoch
    
    # save all targMean, targStd in args since they are important model parameters
    args.targMean = []
    args.targStd = []
    for dat in datasets:
        args.targMean.append(dat.targMean)
        args.targStd.append(dat.targStd)

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
    else:
        print >>log, "No transformation performed on inputs."
        preproInps = lambda xx: xx.toarray()

    # prepare target output activity
    preproTargs = lambda yy: yy
                        
    # print some training details
    print >>log, "%s records PER DATASET per miniBatch" % (','.join(map(str,args.mbsz)))
    print >>log, "%d total records per miniBatch, %d miniBatches per epoch, and %d epochs." % (args.totalmbsz, args.mbPerEpoch, args.epochs) 
    print >>log, "Number of nodes in hidden Layers:" 
    if args.loadModel is None:
        print >>log, args.hid
    else:
        print >>log, "(same as loaded model)"
    if args.useAll:
        print >>log, "Use all training data to build miniBatches in each epoch."
    if args.CV > 0:
        print >>log, "Percentage of leave-out cross-validation data in original training data: %.1f%%" % (100*args.CV)
    if args.watchStr is not None:
        print >>log, "Monitor dataset: %s " % args.watchStr
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
        writePredsSummary(mseCVFile, MSEs_CV, args.datNames)
        rsqCVFile = os.path.join(args.savePrefix, "rsqDuringTraining_CV.csv")
        writePredsSummary(rsqCVFile, RSQs_CV, args.datNames)
    if args.test:
        mseTestFile = os.path.join(args.savePrefix, "MSEDuringTraining_Test.csv")
        writePredsSummary(mseTestFile, MSEs_Test, args.datNames)
        rsqTestFile = os.path.join(args.savePrefix, "rsqDuringTraining_Test.csv")
        writePredsSummary(rsqTestFile, RSQs_Test, args.datNames)
        
    print >>log, "Finish time: %s " % (tstamp())
    _logFile.close()


if __name__ == "__main__":
    main()
