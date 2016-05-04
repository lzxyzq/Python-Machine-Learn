# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 19:34:05 2016

@author: yue
"""
from numpy import *
def loadDataSet():
    dataMat = [];labelMat = [];
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat
    
def sigmoid(inX):
    return 1.0/(1.0+exp(-inX))
    
def gradAscent(dataMatIn,classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m,n=shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights= ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)
        error = (labelMat-h)
        weights = weights + alpha * dataMatrix.transpose()*error
    return weights
def plotBestFit(wei):
    
