#!usr/bin/python
#coding=utf-8
import reader
import cnnConfig as CONFIG
import random
import Utils
import math
import time
import Calc

def getAccuracy(imageList, labelList):
    correctNum = 0
    testLabelList = []
    for i in range(len(imageList)):
        image = imageList[i]
        matrixList = formatInputMatrix(image)
        c1_layer = computeConvolution(matrixList, CONFIG.kernel_i_c1, CONFIG.bias_i_c1)
        s2_layer = computeSample(c1_layer, CONFIG.C1_SIZE / CONFIG.S2_SIZE)
        c3_layer = computeConvolution(s2_layer, CONFIG.kernel_s2_c3, CONFIG.bias_s2_c3)
        s4_layer = computeSample(c3_layer, CONFIG.C3_SIZE / CONFIG.S4_SIZE)
        o_layer = computeConvolution(s4_layer, CONFIG.kernel_s4_o, CONFIG.bias_s4_o)
        outputList = outputLayerToList(o_layer)
        label = Calc.getMaxIndex(outputList)
        testLabelList.append([labelList[i], label])
        if label == labelList[i]:
            correctNum += 1
    print testLabelList
    return float(correctNum) / len(imageList)

def formatInputMatrix(inputMatrix):
    matrixList = [inputMatrix]
    return matrixList

'''
@description cnn训练参数
@param
'''
def trainModel(image, label):
    matrixList = formatInputMatrix(image)
    c1_layer = computeConvolution(matrixList, CONFIG.kernel_i_c1, CONFIG.bias_i_c1)
    s2_layer = computeSample(c1_layer, CONFIG.C1_SIZE / CONFIG.S2_SIZE)
    c3_layer = computeConvolution(s2_layer, CONFIG.kernel_s2_c3, CONFIG.bias_s2_c3)
    s4_layer = computeSample(c3_layer, CONFIG.C3_SIZE / CONFIG.S4_SIZE)
    o_layer = computeConvolution(s4_layer, CONFIG.kernel_s4_o, CONFIG.bias_s4_o)

    o_error = setOutLayerErrors(o_layer, label)
    s4_error = setSampErrors(CONFIG.kernel_s4_o, o_error)
    c3_error = setConvErrors(c3_layer, CONFIG.C3_SIZE / CONFIG.S4_SIZE, s4_error)
    s2_error = setSampErrors(CONFIG.kernel_s2_c3, c3_error)
    c1_error = setConvErrors(c1_layer, CONFIG.C1_SIZE / CONFIG.S2_SIZE, s2_error)

    updateKernels(o_error, CONFIG.kernel_s4_o, s4_layer)
    updateKernels(c3_error, CONFIG.kernel_s2_c3, s2_layer)
    updateKernels(c1_error, CONFIG.kernel_i_c1, matrixList)
    updateBias(o_error, CONFIG.bias_s4_o)
    updateBias(c3_error, CONFIG.bias_s2_c3)
    updateBias(c1_error, CONFIG.bias_i_c1)

'''
@description 设置输出层残差
@param
'''
def setOutLayerErrors(outputLayer, label):
    targetList = [0] * 10
    targetList[label] = 1

    errorLayer = []
    for i in range(len(outputLayer)):
        errorMatrix = []
        for j in range(len(outputLayer[i])):
            errorList = []
            for k in range(len(outputLayer[i][j])):
                outValue = outputLayer[i][j][k]
                error = outValue * (1.0 - outValue) * (targetList[i] - outValue)
                errorList.append(error)
            errorMatrix.append(errorList)
        errorLayer.append(errorMatrix)
    return errorLayer

'''
@description 设置采样层的残差
@param
'''
def setSampErrors(kernel, nextErrorLayer):
    #printSize(kernel)
    currentOutputNumber = len(kernel[0])
    nextOutputNumber = len(kernel)
    errorList = []
    for i in range(currentOutputNumber):
        sumLayer = None
        for j in range(nextOutputNumber):
            nextErrorMatrix = nextErrorLayer[j]
            kernelRot = Calc.matrixRot180(kernel[j][i])
            if sumLayer == None:
                sumLayer = Calc.convnFull(nextErrorMatrix, kernelRot)
            else:
                sumLayer = Calc.matrixOp2(
                    Calc.convnFull(nextErrorMatrix, kernelRot),
                    sumLayer,
                    None,
                    None,
                    Calc.plus)
        errorList.append(sumLayer)
    return errorList

def multiplyAlpha(a):
    return a * CONFIG.RATE

def updateKernels(errorLayer, kernels, prevLayer):
    for j in range(len(kernels)):# 10
        for i in range(len(kernels[0])):# 12
            deltaKernel = None
            error = errorLayer[j]
            if deltaKernel == None:
                deltaKernel = Calc.convnValid(prevLayer[i], error)
            else:
                deltaKernel = Calc.matrixOp2(deltaKernel,
                    Calc.convnValid(prevLayer[i], error),
                    None,
                    None,
                    Calc.plus)

            kernel = kernels[j][i] 
            kernels[j][i] = Calc.matrixOp2(deltaKernel, kernel, multiplyAlpha, None, Calc.plus)

def updateBias(errorLayer, biasList):
    for i in range(len(biasList)):
        error = errorLayer[i]
        deltaBias = Calc.sum(error)
        biasList[i] += CONFIG.RATE * deltaBias

'''
@description 设置卷积层的残差
@param
'''
def setConvErrors(currentLayer, sampleKernelSize, nextErrorLayer):
    errorList = []
    for i in range(len(nextErrorLayer)):
        nextError = nextErrorLayer[i]
        layer = currentLayer[i]
        outMatrix = Calc.matrixOp2(layer, Calc.cloneMatrix(layer), None, Calc.oneValue, Calc.multiply)
        outMatrix = Calc.matrixOp2(outMatrix,
            Calc.kronecker(nextError, sampleKernelSize, sampleKernelSize),
            None, None, Calc.multiply)
        errorList.append(outMatrix)
    return errorList


'''
转换输出结果为一维列表
'''
def outputLayerToList(m):
    list = []
    for i in range(len(m)):
        for j in range(len(m[i])):
            for k in range(len(m[i][j])):
                list.append(m[i][j][k])
    return list

'''
计算卷积层
有问题，卷积核的个数应该是 前个数*后个数*kernel
'''
def computeConvolution(matrixList, kernelMatrixMatrix, biasList):
    # Calc.printMatrix(inputMatrix)
    resultList = []
    for i in range(len(kernelMatrixMatrix)):
        size = len(matrixList[0]) - len(kernelMatrixMatrix[i][0]) + 1
        newMatrix = Utils.nList(size, size)
        kernelMatrixList = kernelMatrixMatrix[i]
        bias = biasList[i]
        for j in range(len(kernelMatrixList)):
            resultMatrix = Calc.convnValid(matrixList[j], kernelMatrixList[j])
            newMatrix = Calc.matrixOp2(newMatrix, resultMatrix, None, None, Calc.plus)
        newMatrix = Calc.matrixAddValue(newMatrix, bias)
        newMatrix = Calc.matrixOp(newMatrix, Calc.sigmod)
        resultList.append(newMatrix)
        # print '%d->%d,%d' % (i, len(newMatrix), len(newMatrix[0]))
    return resultList

def computeSample(inputMatrixList, kernelSize):
    resultList = []
    for i in range(len(inputMatrixList)):
        inputMatrix = inputMatrixList[i]
        resultMatrix = Calc.scaleMatrix(inputMatrix, kernelSize, kernelSize)
        resultList.append(resultMatrix)
    return resultList

def printSize(matrix):
    print '%d,%d' % (len(matrix), len(matrix[0]))

'''
@description 读取文件中的权值参数
'''
def readResultFromFile():
    return (reader.readListFromFile(CONFIG.KERNEL_I_C1_FILE, CONFIG.kernel_i_c1)
        and reader.readListFromFile(CONFIG.KERNEL_S2_C3_FILE, CONFIG.kernel_s2_c3)
        and reader.readListFromFile(CONFIG.KERNEL_S4_O_FILE, CONFIG.kernel_s4_o)
        and reader.readListFromFile(CONFIG.BIAS_I_C1, CONFIG.bias_i_c1)
        and reader.readListFromFile(CONFIG.BIAS_S2_C3, CONFIG.bias_s2_c3)
        and reader.readListFromFile(CONFIG.BIAS_S4_O, CONFIG.bias_s4_o))

def initWeightAndBias():
    fromFileSuccess = readResultFromFile()
    if fromFileSuccess:
        print '从文件中初始化权值参数成功'
        return

'''
@description 保存训练后的参数到文件
'''
def saveResultToFile():
    reader.saveList(CONFIG.KERNEL_I_C1_FILE, CONFIG.kernel_i_c1)
    reader.saveList(CONFIG.KERNEL_S2_C3_FILE, CONFIG.kernel_s2_c3)
    reader.saveList(CONFIG.KERNEL_S4_O_FILE, CONFIG.kernel_s4_o)

    reader.saveList(CONFIG.BIAS_I_C1, CONFIG.bias_i_c1)
    reader.saveList(CONFIG.BIAS_S2_C3, CONFIG.bias_s2_c3)
    reader.saveList(CONFIG.BIAS_S4_O, CONFIG.bias_s4_o)


if __name__ == '__main__':

    #初始参数
    imageNumber = 60000

    #初始化权值矩阵等
    initWeightAndBias()

    #读取label数据
    labelReader = reader.BitFileReader()
    labelReader.open(CONFIG.LABEL_FILE)
    labelReader.step(8)
    labelList = []
    for i in range(imageNumber):
        labelList.append(labelReader.read())
    labelReader.close()

    #读取图片数据
    imageReader = reader.BitFileReader()
    imageReader.open(CONFIG.TRAIN_FILE)
    imageReader.step(16) #由于前面的都是无用数据，直接跳到16这个位置

    imageList = []
    for i in range(imageNumber):
        imageList.append(Utils.readImageLinear(imageReader))
    imageReader.close()

    # 打印读取的数据
    # for i in range(imageNumber):
    #     Utils.printImage(imageList[i])
    #     print labelList[i]
    accuracy = getAccuracy(imageList, labelList)
    print '初始准确率：%s\n' % str(accuracy)

    trainTimes = 0
    startTime = time.time()
    while True:
        trainTimes += 1
        if accuracy > CONFIG.THRESHOD or trainTimes > CONFIG.TRAIN_TIMES:
            break
        for i in range(len(imageList)):
            trainModel(imageList[i], labelList[i])

        if trainTimes % 10 == 0 or True:
            accuracy = getAccuracy(imageList, labelList)
            print '第%d次训练，准确率：%f' % (trainTimes, accuracy)
            print '耗时%fs' % (time.time() - startTime)

    while True:
        print '是否需要保存权重等数据到文件中？(Y/N)'
        needSave = raw_input()
        if needSave.upper() == 'Y':
            saveResultToFile()
            print '保存成功，训练结束'
            break
        elif needSave.upper() == 'N':
            print '训练结束'
            break