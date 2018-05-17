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
Simple feed forward neural network key components.

This code is developed based on George Dahl and Junshui Ma's code [dbn.py].
All pre-training procedures are removed.
Last modified by Yuting Xu on Feb. 08, 2017.
"""

import numpy as num
import gnumpy as gnp
import itertools
from activationFunctions import *
from counter import Progress


def initWeightMatrix(shape, scale, maxNonZeroPerColumn = None, uniform = False):
    """
    Generate initial weight matrix according to uniform or gaussian distribution
    """
    # fanIn is the number of nonzero incoming connections to a hidden unit
    fanIn = shape[0] if maxNonZeroPerColumn==None else min(maxNonZeroPerColumn, shape[0])
    if uniform:
        W = scale*(2*num.random.rand(*shape)-1)
    else:
        W = scale*num.random.randn(*shape)
    # randomly set (nrow-fanIn) weights to 0 (WHY?)
    for j in range(shape[1]):
        perm = num.random.permutation(shape[0])
        W[perm[fanIn:],j] *= 0
    return W

def garrayify(arrays):
    """
    convert other arrays into gpu-array
    """
    return [ar if isinstance(ar, gnp.garray) else gnp.garray(ar) for ar in arrays]

def numpyify(arrays):
    """
    convert other structures into numpy arrays
    """
    return [ar if isinstance(ar, num.ndarray) else ar.as_numpy_array(dtype=num.float32) for ar in arrays]

def limitColumnRMS(W, rmsLim):
    """
    All columns of W with rms entry above the limit are scaled to equal the limit.
    The limit can either be a row vector or a scalar.
    Apply to 2-d array W.
    """
    columnRMS = lambda W: gnp.sqrt(gnp.mean(W*W,axis=0))
    rmsScale = rmsLim/columnRMS(W)
    return W*(1 + (rmsScale < 1)*(rmsScale-1))

def loadSavedNeuralNet(path, dense = False):
    """
    load a saved model, which maybe the output from previous training process,
    and return a DNN object build from parameters in this model
    """
    d = num.load(path)
    layerSizes = d['layerSizes']
    outputActFunct = d['outputActFunct']
    useReLU = d['useReLU']
    weights = garrayify(d['weights'].flatten())
    biases = garrayify(d['biases'].flatten())
    targMean = d['targMean'] 
    targStd = d['targStd']

    if not dense:
        net = DNN(layerSizes, LinearMasked(), useReLU, weights, biases, targMean, targStd)
    else:
        net = DNN(layerSizes, Linear(), useReLU, weights, biases, targMean, targStd)

    return net, d

    
class DNN(object):
    def __init__(self, layerSizes=None, outputActFunct=Linear(), useReLU = True, \
                 initialWeights=None, initialBiases=None, targMean=None, targStd=None):
        """
        Construct a Neural Network object with Basic Structure:
         - layerSizes: [input size, hidden layer size list, output size]
         - outputActFunct: activation function for output layer, such as Linear() and LinearMasked()
         - useReLU: True/False, use ReLU() or Sigmoid() as activation function
        """
        self.layerSizes = layerSizes
        self.outputActFunct = outputActFunct
        self.useReLU = useReLU
        
        if useReLU:
            self.hidActFuncts = [ReLU() for i in range(len(layerSizes) - 2)]
        else:
            self.hidActFuncts = [Sigmoid() for i in range(len(layerSizes) - 2)]

        # initialize weights and biases
        if initialWeights is None:
			# set wscale for each layer according to 0.5*n*Var(w) = 1
            scale_list = [num.sqrt(2.0/n) for n in layerSizes[:-1]]
            shapes = [(layerSizes[i-1],layerSizes[i]) for i in range(1, len(layerSizes))]
            self.weights = [gnp.garray(initWeightMatrix(shapes[i], scale_list[i], None, False)) for i in range(len(shapes))]
        else:
            self.weights = initialWeights

        if initialBiases is None:
            self.biases = [gnp.garray(0*num.random.rand(1, self.layerSizes[i])) for i in range(1, len(self.layerSizes))]
        else:
            self.biases = initialBiases
        
        # initialize gradients of weights and biases
        self.WGrads = [gnp.zeros(self.weights[i].shape) for i in range(len(self.weights))]
        self.biasGrads = [gnp.zeros(self.biases[i].shape) for i in range(len(self.biases))]

        # specify targMean and targStd with model since they are important model parameters
        #assert(len(targMean) == layerSizes[-1])
        self.targMean = targMean
        #assert(len(targStd) == layerSizes[-1])
        self.targStd = targStd

    def NeuralNetParametersDict(self):
        """
        After training, collect all parameters in a dictionary to save. 
        """
        d = vars(self)
        if 'WGrads' in d: del d['WGrads']
        if 'biasGrads' in d: del d['biasGrads']
        if 'state' in d: del d['state']
        if 'acts' in d: del d['acts']
        
        if len(self.weights) == 1:
            d['weights'] = num.empty((1,), dtype=num.object)
            d['weights'][0] = numpyify(self.weights)[0]
            d['biases'] = num.empty((1,), dtype=num.object)
            d['biases'][0] = numpyify(self.biases)[0]
        else:
            d['weights'] = num.array(numpyify(self.weights)).flatten()
            #d['biases'] = num.array(numpyify(self.biases)).flatten()
            d['biases'] = num.array([bb.flatten() for bb in numpyify(self.biases)])
        d['outputActFunct'] = self.outputActFunct.__class__.__name__
        return d
        
    def scaleDerivs(self, scale):
        """
        Multiply all weights and bias gradients by a constant momentum
        Used by another class function [Step].
        """
        for i in range(len(self.weights)):
            self.WGrads[i] *= scale
            self.biasGrads[i] *= scale

    def fineTune(self, minibatchStream, epochs, mbPerEpoch, loss = None, progressBar = True, useDropout = False):
        for ep in range(epochs):
            totalCases = 0
            sumErr = 0
            sumLoss = 0
            if self.nesterov:
                step = self.stepNesterov
            else:
                step = self.step
            prog = Progress(mbPerEpoch) if progressBar else DummyProgBar()
            for i in range(mbPerEpoch):
                if isinstance(self.outputActFunct, LinearMasked):
                    inpMB, targMB, targMaskMB = minibatchStream.next()
                    err, outMB = step(inpMB, targMB, self.learnRates, self.momentum, self.L2Costs, useDropout, targMaskMB)
                else:
                    inpMB, targMB = minibatchStream.next()
                    err, outMB = step(inpMB, targMB, self.learnRates, self.momentum, self.L2Costs, useDropout)
                sumErr += err
                if loss != None:
                    sumLoss += loss(targMB, outMB)
                totalCases += inpMB.shape[0]
                prog.tick()
            prog.done()
            yield sumErr/float(totalCases), sumLoss/float(totalCases)
    
    def totalLoss(self, minibatchStream, lossFuncts):
        totalCases = 0
        sumLosses = num.zeros((1+len(lossFuncts),))
        if isinstance(self.outputActFunct, LinearMasked):
            for inpMB, targMB, targMaskMB in minibatchStream:
                inputBatch = inpMB if isinstance(inpMB, gnp.garray) else gnp.garray(inpMB)
                targetBatch = targMB if isinstance(targMB, gnp.garray) else gnp.garray(targMB)
                targetMaskBatch = targMaskMB if isinstance(targMaskMB, gnp.garray) else gnp.garray(targMaskMB)
            
                outputActs = self.fprop(inputBatch)
                sumLosses[0] += self.outputActFunct.error(targetBatch, self.state[-1], targetMaskBatch, outputActs)
                for j,f in enumerate(lossFuncts):
                    sumLosses[j+1] += f(targetBatch, outputActs, targetMaskBatch)
                totalCases += inpMB.shape[0]
        else:
            for inpMB, targMB in minibatchStream:
                inputBatch = inpMB if isinstance(inpMB, gnp.garray) else gnp.garray(inpMB)
                targetBatch = targMB if isinstance(targMB, gnp.garray) else gnp.garray(targMB)
    
                outputActs = self.fpropDropout(inputBatch)
                sumLosses[0] += self.outputActFunct.error(targetBatch, self.state[-1], outputActs)
                for j,f in enumerate(lossFuncts):
                    sumLosses[j+1] += f(targetBatch, outputActs)
                totalCases += inpMB.shape[0]
        return sumLosses / float(totalCases)

    def predictions(self, minibatchStream, asNumpy = False, useDropout = False):
        """
        Perform prediction with option to use drop-out or not.
        Used by [DeepNeuralNetPredict.py]
        """
        for inpMB in minibatchStream:
            inputBatch = inpMB if isinstance(inpMB, gnp.garray) else gnp.garray(inpMB)
            outputActs = self.fpropDropout(inputBatch, useDropout)
            yield outputActs.as_numpy_array() if asNumpy else outputActs

    def fpropBprop(self, inputBatch, targetBatch, useDropout, targetMaskBatch = None):
        outputActs = self.fpropDropout(inputBatch, useDropout)

        if (targetMaskBatch == None):
            outputErrSignal = -self.outputActFunct.dErrordNetInput(targetBatch, self.state[-1], outputActs)
            error = self.outputActFunct.error(targetBatch, self.state[-1], outputActs)
        else:
            outputErrSignal = -self.outputActFunct.dErrordNetInput(targetBatch, self.state[-1], targetMaskBatch, outputActs)
            error = self.outputActFunct.error(targetBatch, self.state[-1], targetMaskBatch, outputActs)

        errSignals = self.bprop(outputErrSignal)
        return errSignals, outputActs, error
    
    def constrainWeights(self):
        for i in range(len(self.rmsLims)):
            if self.rmsLims[i] != None:
                self.weights[i] = limitColumnRMS(self.weights[i], self.rmsLims[i])
    
    def step(self, inputBatch, targetBatch, learnRates, momentum, L2Costs, useDropout = False, targetMaskBatch = None):
        mbsz = inputBatch.shape[0]
        inputBatch = inputBatch if isinstance(inputBatch, gnp.garray) else gnp.garray(inputBatch)
        if (targetMaskBatch is None):
            targetBatch = targetBatch if isinstance(targetBatch, gnp.garray) else gnp.garray(targetBatch)
            errSignals, outputActs, error = self.fpropBprop(inputBatch, targetBatch, useDropout)
        else:
            targetMaskBatch = targetMaskBatch if isinstance(targetMaskBatch, gnp.garray) else gnp.garray(targetMaskBatch)
            errSignals, outputActs, error = self.fpropBprop(inputBatch, targetBatch, useDropout, targetMaskBatch)

        factor = 1-momentum if not self.nestCompare else 1.0
        self.scaleDerivs(momentum)
        for i, (WGrad, biasGrad) in enumerate(self.gradients(self.state, errSignals)):
            self.WGrads[i] += learnRates[i]*factor*(WGrad/mbsz - L2Costs[i]*self.weights[i])
            self.biasGrads[i] += (learnRates[i]*factor/mbsz)*biasGrad
        self.applyUpdates(self.weights, self.biases, self.weights, self.biases, self.WGrads, self.biasGrads)
        self.constrainWeights()
        return error, outputActs
   
    def stepNesterov(self, inputBatch, targetBatch, learnRates, momentum, L2Costs, useDropout = False):
        mbsz = inputBatch.shape[0]
        inputBatch = inputBatch if isinstance(inputBatch, gnp.garray) else gnp.garray(inputBatch)
        targetBatch = targetBatch if isinstance(targetBatch, gnp.garray) else gnp.garray(targetBatch)
        if isinstance(self.outputActFunct, LinearMasked):
            targetMaskBatch = targetMaskBatch if isinstance(targetMaskBatch, gnp.garray) else gnp.garray(targetMaskBatch)

        curWeights = [w.copy() for w in self.weights]
        curBiases = [b.copy() for b in self.biases]
        self.scaleDerivs(momentum)
        self.applyUpdates(self.weights, self.biases, curWeights, curBiases, self.WGrads, self.biasGrads)

        if isinstance(self.outputActFunct, LinearMasked):
            errSignals, outputActs, error = self.fpropBprop(inputBatch, targetBatch, useDropout, targetMaskBatch)
        else:
            errSignals, outputActs, error = self.fpropBprop(inputBatch, targetBatch, useDropout)
        
        #self.scaleDerivs(momentum)
        for i, (WGrad, biasGrad) in enumerate(self.gradients(self.state, errSignals)):
            self.WGrads[i] += learnRates[i]*(WGrad/mbsz - L2Costs[i]*self.weights[i])
            self.biasGrads[i] += (learnRates[i]/mbsz)*biasGrad
        self.applyUpdates(self.weights, self.biases, curWeights, curBiases, self.WGrads, self.biasGrads)
        self.constrainWeights()
        return error, outputActs

    def applyUpdates(self, destWeights, destBiases, curWeights, curBiases, WGrads, biasGrads):
        for i in range(len(destWeights)):
            destWeights[i] = curWeights[i] + WGrads[i]
            destBiases[i] = curBiases[i] + biasGrads[i]
    
    def fpropDropout(self, inputBatch, useDropout = False, weightsToStopBefore = None):
        """
        Perform a (possibly partial) forward pass through the
        network. Updates self.state which, on a full forward pass,
        holds the input followed by each hidden layer's activation and
        finally the net input incident on the output layer. For a full
        forward pass, we return the actual output unit activations. In
        a partial forward pass we return None.
        If useDropout == True, ranomly drop units for each layer. 
        """
        inputBatch = inputBatch if isinstance(inputBatch, gnp.garray) else gnp.garray(inputBatch)
        if weightsToStopBefore == None:
            weightsToStopBefore = len(self.weights)
        self.keptMask = [gnp.rand(*inputBatch.shape) > self.dropouts[0]]
        #self.state holds everything before the output nonlinearity, including the net input to the output units
        self.state = [inputBatch * self.keptMask[0]]
        for i in range(min(len(self.weights) - 1, weightsToStopBefore)):
            if useDropout:
                dropoutMultiplier = 1.0/(1.0-self.dropouts[i])
                curActs = self.hidActFuncts[i].activation(gnp.dot(dropoutMultiplier*self.state[-1], self.weights[i]) + self.biases[i])
                self.keptMask.append(gnp.rand(*curActs.shape) > self.dropouts[i+1])
                self.state.append(curActs * self.keptMask[-1])
            else:
                curActs = self.hidActFuncts[i].activation(gnp.dot(self.state[-1], self.weights[i]) + self.biases[i])
                self.state.append(curActs)
        if weightsToStopBefore >= len(self.weights):
            if useDropout:
                dropoutMultiplier = 1.0/(1.0-self.dropouts[-1])
                self.state.append(gnp.dot(dropoutMultiplier*self.state[-1], self.weights[-1]) + self.biases[-1])
            else:
                self.state.append(gnp.dot(self.state[-1], self.weights[-1]) + self.biases[-1])                
            self.acts = self.outputActFunct.activation(self.state[-1])
            return self.acts
        # If we didn't reach the output units
        # To return the first set of hidden activations, we would set
        # weightsToStopBefore to 1.
        return self.state[weightsToStopBefore]

    def bprop(self, outputErrSignal, fpropState = None):
        """
        Perform a backward pass through the network. fpropState
        defaults to self.state (set during fprop) and outputErrSignal
        should be self.outputActFunct.dErrordNetInput(...).
        """
        #if len(errSignals)==len(self.weights)==len(self.biases)==h+1 then
        # len(fpropState) == h+2 because it includes the input and the net input to the output layer and thus
        #fpropState[-2] is the activation of the penultimate hidden layer (or the input if there are no hidden layers)
        if fpropState == None:
            fpropState = self.state
        assert(len(fpropState) == len(self.weights) + 1)

        errSignals = [None for i in range(len(self.weights))]
        errSignals[-1] = outputErrSignal
        for i in reversed(range(len(self.weights) - 1)):
            errSignals[i] = gnp.dot(errSignals[i+1], self.weights[i+1].T)*self.hidActFuncts[i].dEdNetInput(fpropState[i+1])
        return errSignals

    def gradients(self, fpropState, errSignals):
        """
        Lazily generate (negative) gradients for the weights and biases given
        the result of fprop (fpropState) and the result of bprop
        (errSignals).
        """
        assert(len(fpropState) == len(self.weights)+1)
        assert(len(errSignals) == len(self.weights) == len(self.biases))
        for i in range(len(self.weights)):
            yield gnp.dot(fpropState[i].T, errSignals[i]), errSignals[i].sum(axis=0)
    

