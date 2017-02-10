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

import numpy as num
import gnumpy as gnp

#NOTATION:
#we use y_l for the output of layer l
#y_0 is input
#
#we use x_l for the net input so, using * as matrix multiply and h_l
#for the elementwise activation function of layer l,
#x_l = y_{l-1} * W_l + b_l
#y_l = h_l(x_l)
#
#A neural net with L layers implements the function f(y_0, W) = y_L where
#y_0 is the input to the network and W represents all of the weights
#and biases of the network.
#We train neural nets to minimize some error function 
# error(y, t) for fixed targets t. 
#So given training inputs y_0 and targets t we minimize the function
#Error(W) = error( f(y_0, W), t)
#
#An activation function suitable for use as a hidden layer
#nonlinearity defines the following methods:
# 1A. activation(netInput)
# 2A. dEdNetInput(acts) 
# 
#An activation function suitable for use as the output layer
#nonlinearity defines the following methods in addiction to 1A:
# 1B. error(targets, netInput, acts = None)
# 2B. dErrordNetInput(targets, netInput, acts = None)
# 3.  HProd(vect, acts)
#
# 1B takes as an argument the net input to the output units because
# sometimes having that quantity allows the loss to be computed in a
# more numerically stable way. Optionally, 1B also takes the output
# unit activations, since sometimes that allows a more efficient
# computation of the loss.
#
# For using featureImportance with dropout, an errorEachCase method is
# also needed. The error method can generally be implemented by
# calling the errorEachCase method.
#
# For "matching" error functions and output activation functions 2B
# should be just acts-targets.
# The difference between 2B and 2A (above) is that 2B incorporates the
# training criterion error(y,t) instead of just the error *at the
# output of this layer* the way 2A does.
#
# HProd gives the product of the H_{L,M} Hessian (Notation from "Fast
# Curvature Matrix-Vector Products for Second-Order Gradient Descent
# by N. Schraudolph) with a vector. 

#If gnumpy gets replaced and a logOnePlusExp is needed, be sure to make it numerically stable.
#def logOnePlusExp(x):
#    # log(1+exp(x)) when x < 0 and
#    # x + log(1+exp(-x)) when x > 0


class Sigmoid(object):
    def activation(self, netInput):
        return netInput.sigmoid()
    def dEdNetInput(self, acts):
        return acts*(1-acts)
    def errorEachCase(self, targets, netInput, acts = None):
        return (netInput.log_1_plus_exp()-targets*netInput).sum(axis=1)
    def error(self, targets, netInput, acts = None):
        #return (targets*logOnePlusExp(-netInput) + (1-targets)*logOnePlusExp(netInput)).sum()
        #return (logOnePlusExp(netInput)-targets*netInput).sum()
        #return (netInput.log_1_plus_exp()-targets*netInput).sum()
        return self.errorEachCase(targets, netInput, acts).sum()
    def HProd(self, vect, acts):
        return vect*acts*(1-acts)
    def dErrordNetInput(self, targets, netInput, acts = None):
        if acts == None:
            acts = self.activation(netInput)
        return acts - targets

#You can write tanh in terms of sigmoid.
#def tanh(ar):
#    return 2*(2*ar).sigmoid()-1
# There might be a "better" tanh to use based on Yann LeCun's
# efficient backprop paper, but I forget what the constants A and B
# are in A * tanh ( B * x).
class Tanh(object):
    def activation(self, netInput):
        return gnp.tanh(netInput)
    def dEdNetInput(self, acts):
        return 1-acts*acts

class ReLU(object):
    def activation(self, netInput):
        return netInput*(netInput > 0)
    def dEdNetInput(self, acts):
        return acts > 0

class Linear(object):
    def activation(self, netInput):
        return netInput
    def dEdNetInput(self, acts):
        return 1 #perhaps returning ones(acts.shape) is more appropriate?
    def errorEachCase(self, targets, netInput, acts = None):
        diff = targets-netInput
        return 0.5*(diff*diff).sum(axis=1)
    def error(self, targets, netInput, acts = None):
        #diff = targets-netInput
        #return 0.5*(diff*diff).sum()
        return self.errorEachCase(targets, netInput, acts).sum()
    def HProd(self, vect, acts):
        return vect
    def dErrordNetInput(self, targets, netInput, acts = None):
        if acts == None:
            acts = self.activation(netInput)
        return acts - targets

class LinearMasked(object):
    """
    For multi-task DNN
    """
    def activation(self, netInput, mask = None):
        if mask == None:
            return netInput
        return netInput*mask
    def errorEachCase(self, targets, netInput, mask, acts = None):
        # WHY?
#        diff = (targets-netInput)*mask
        diff = (targets-netInput.as_numpy_array())*mask.as_numpy_array()
        return 0.5*(diff*diff).sum(axis=1)
    def error(self, targets, netInput, mask, acts = None):
        #diff = targets-netInput
        #return 0.5*(diff*diff).sum()
        return self.errorEachCase(targets, netInput, mask, acts).sum()
    def HProd(self, vect, acts):
        raise NotImplementedError()
    def dErrordNetInput(self, targets, netInput, mask, acts = None):
        if acts == None:
            acts = self.activation(netInput, mask)
        return (acts - targets)*mask

class Softmax(object):
    def activation(self, netInput):
        Zshape = (netInput.shape[0],1)
        acts = netInput - netInput.max(axis=1).reshape(*Zshape)
        acts = acts.exp()
        return acts/acts.sum(axis=1).reshape(*Zshape)
    def HProd(self, vect, acts):
        return acts*(vect-(acts*vect).sum(1).reshape(-1,1))
    def dErrordNetInput(self, targets, netInput, acts = None):
        if acts == None:
            acts = self.activation(netInput)
        return acts - targets
    def errorEachCase(self, targets, netInput, acts = None):
        ntInpt = netInput - netInput.max(axis=1).reshape(netInput.shape[0],1)
        logZs = ntInpt.exp().sum(axis=1).log().reshape(-1,1)
        err = targets*(ntInpt - logZs)
        return -err.sum(axis=1)
    def error(self, targets, netInput, acts = None):
        #ntInpt = netInput - netInput.max(axis=1).reshape(netInput.shape[0],1)
        #logZs = ntInpt.exp().sum(axis=1).log().reshape(-1,1)
        #err = targets*(ntInpt - logZs)
        #return -err.sum()
        return self.errorEachCase(targets, netInput, acts).sum()






