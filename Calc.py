#!usr/bin/python
#coding=utf-8
import math
import copy
import Utils

# 1-value
def oneValue(value):
    return 1.0 - value

def digmod(value):
    return 1 / (1.0 + math.exp(-value))

def plus(a, b):
    return a + b

def multiply(a, b):
    return a * b

def minus(a, b):
    return a - b

'''
@description 打印矩阵
@param
'''
def printMatrix(matrix):
    for i in range(len(matrix)):
        print matrix[i]
    print ''

'''
@description 矩阵旋转180度啊
@param
'''
def matrixRot180(matrix):
    nMatrix = copy.deepcopy(matrix)
    m = len(nMatrix)
    n = len(nMatrix[0])
    for i in range(m):
        for j in range(n / 2):
            t = nMatrix[i][j]
            nMatrix[i][j] = nMatrix[i][n - 1 - j]
            nMatrix[i][n - 1 - j] = t

    for j in range(n):
        for i in range(m / 2):
            t = nMatrix[i][j]
            nMatrix[i][j] = nMatrix[m - 1 - i][j]
            nMatrix[m - 1 - i][j] = t
    return nMatrix

'''
@description 初始化二维列表
@param {int} x
@param {int} y
'''
def randomInitMatrix(x, y):
    matrix = Utils.nList(x, y)
    tag = 1
    for i in range(x):
        for j in range(y):
            matrix[i][j] = (random.randint(0, 1000) - 500) / 10000.0
    return matrix

'''
@description 初始化列表
@param {int} len
'''
def randomInitList(len):
    list = []
    for i in range(len):
        list.append(0)
    return list

'''
@description 随机排列的抽样，随机抽取batchSize个[0,size)的数
@param
'''
def randomPerm(size, batchSize):
    newSet = []
    for i in range(batchSize):
        newSet.append(random.randint(0, size))

    return newSet

'''
@description 复制矩阵
@param
'''
def cloneMatrix(matrix):
    newMatrix = []
    for i in range(len(matrix)):
        newList = []
        for j in range(len(matrix[i])):
            newList.append(matrix[i][j])
        newMatrix.append(newList)
    return newMatrix

def matrixAddValue(matrix, value):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            matrix[i][j] += value
    return matrix

'''
@description 对单个矩阵进行操作
@param
'''
def matrixOp(matrix, operator):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            matrix[i][j] = operator(matrix[i][j])
    return matrix

'''
@description 两个维度相同的矩阵对应元素操作,得到的结果方法mb中，即mb[i][j] = (op_a * ma[i][j]) op (op_b mb[i][j])
@param
'''
def matrixOp2(matrixA, matrixB, operatorA, operatorB, operatorOnTwo):
    if len(matrixA) != len(matrixB) or len(matrixA[0]) != len(matrixB[0]):
        print 'matrixOp.error:->两个矩阵大小不一致 len(matrixA): %s, len(matrixB): %s' % (len(matrixA), len(matrixB))

    for i in range(len(matrixA)):
        for j in range(len(matrixA[i])):
            a = matrixA[i][j]
            if operatorA:
                a = operatorA(a)
            b = matrixB[i][j]
            if operatorB:
                b = operatorB(b)
            matrixB[i][j] = operatorOnTwo(a, b)

    return matrixB

'''
@description 克罗内克积,对矩阵进行扩展
@param
@param {int} scaleW
@param {int} scaleH
'''
def kronecker(matrix, scaleW, scaleH):
    outMatrix = Utils.nList(len(matrix * scaleW), len(matrix[0]) * scaleH)
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            for ki in range(i * scaleW, (i + 1) * scaleW):
                for kj in range(j * scaleH, (j + 1) * scaleH):
                    outMatrix[ki][kj] = matrix[i][j]
    return outMatrix

'''
@description 对矩阵进行均值缩小
@param
@param {int} scaleW
@param {int} scaleH
'''
def scaleMatrix(matrix, scaleW, scaleH):
    m = len(matrix)
    n = len(matrix[0])
    sm = m / scaleW
    sn = n / scaleH
    outMatrix = Utils.nList(sm, sn)
    if sm * scaleW != m or sn * scaleH != n:
        print 'scaleMatrix.error->scale不能整除matrix'
    size = scaleW * scaleH
    for i in range(sm):
        for j in range(sn):
            sum = 0.0
            for si in range(i * scaleW, (i + 1) * scaleW):
                for sj in range(j * scaleH, (j + 1) * scaleH):
                    sum += matrix[si][sj]
            outMatrix[i][j] = sum / size

    return outMatrix

'''
@description 计算full模式的卷积
@param
@param {int} scaleW
'''
def convnFull(matrix, kernel):
    m = len(matrix)
    n = len(matrix[0])
    km = len(kernel)
    kn = len(kernel[0])
    extendMatrix = Utils.nList(m + 2 * (km - 1), n + 2 * (kn - 1))
    for i in range(m):
        for j in range(n):
            extendMatrix[i + km - 1][j + kn - 1] = matrix[i][j]
    return convnValid(extendMatrix, kernel)

'''
@description 计算valid模式的卷积
@param
@param {int} scaleW
'''
def convnValid(matrix, kernel):
    m = len(matrix)
    n = len(matrix[0])
    km = len(kernel)
    kn = len(kernel[0])
    # 需要做卷积的列数
    kns = n - kn + 1;
    # 需要做卷积的行数
    kms = m - km + 1;
    # 结果矩阵
    outMatrix = Utils.nList(kms, kns)

    for i in range(kms):
        for j in range(kns):
            sum = 0.0
            for ki in range(km):
                for kj in range(kn):
                    sum += matrix[i + ki][j + kj] * kernel[ki][kj]
            outMatrix[i][j] = sum
    return outMatrix

def sigmod(value):
    if value < -700:
        return 0
    return 1 / (1.0 + math.exp(-value))

'''
@description 对矩阵元素求和
@param
@param {int} scaleW
'''
def sum(errorMatrix):
    sum = 0.0
    for i in range(len(errorMatrix)):
        for j in range(len(errorMatrix[i])):
            sum += errorMatrix[i][j]
    return sum

def listToString(inList):
    s = ''
    for i in range(len(inList)):
        s += str(inList[i])
    return s

'''
@description [1,0,1] => 5
@param
'''
def binaryArray2int(bList):
    newList = Utils.nList(len(bList))
    for i in range(len(newList)):
        if bList[i] >= 0.500000001:
            newList[i] = 1
        else:
            newList[i] = 0
    binaryStr = listToString(newList)
    data = int(binaryStr, 2)
    return data

'''
@description 取最大的元素的下标
@param
'''
def getMaxIndex(out):
    maxValue = out[0]
    index = 0
    for i in range(len(out)):
        if out[i] > maxValue:
            maxValue = out[i]
            index = i
    return index

def fomart(data):
    'mark'
    'pass'


'''
@description 测试卷积,测试结果：4核下并发并行的卷积提高不到2倍
@param
'''
def testConvn():
    count = 1
    m = Utils.nList(5, 5)
    for i in range(len(m)):
        for j in range(len(m[i])):
            m[i][j] = count
            count += 1
    k = Utils.nList(5, 5)
    for i in range(len(k)):
        for j in range(len(k[i])):
            k[i][j] = 1
    printMatrix(m)
    out = convnFull(m, k)
    printMatrix(out)

def testScaleMatrix():
    print 'testScaleMatrix:'
    count = 1
    m = Utils.nList(16, 16)
    for i in range(len(m)):
        for j in range(len(m[i])):
            m[i][j] = count
            count += 1
    out = scaleMatrix(m, 2, 2)
    printMatrix(m)
    printMatrix(out)

def testKronecker():
    print 'testKronecker:'
    count = 1
    m = Utils.nList(5, 5)
    for i in range(len(m)):
        for j in range(len(m[i])):
            m[i][j] = count
            count += 1
    out = kronecker(m, 2, 2)
    printMatrix(m)
    printMatrix(out)

def testOp1(value):
    return value - 1

def testOp2(value):
    return -1 * value

def testMatrixProduct():
    print 'testMatrixProduct:'
    count = 1
    m = Utils.nList(5, 5)
    for i in range(len(m)):
        for j in range(len(m[i])):
            m[i][j] = count
            count += 1

    k = Utils.nList(5, 5)
    for i in range(len(k)):
        for j in range(len(k[i])):
            k[i][j] = j

    printMatrix(m)
    printMatrix(k)
    out = matrixOp2(m, k, testOp1, testOp2, multiply)
    printMatrix(out)
    
def testCloneMatrix():
    print 'testCloneMatrix:'
    count = 1
    m = Utils.nList(5, 5)
    for i in range(len(m)):
        for j in range(len(m[i])):
            m[i][j] = count
            count += 1

    out = cloneMatrix(m)
    printMatrix(m)
    printMatrix(out)

def testRot180():
    matrix = [[1,2,3,4], [4,5,6,7], [7,8,9,10]]
    printMatrix(matrix)
    matrixRot180(matrix)
    printMatrix(matrix)

def testConvnValid():
    print 'testConvnValid'
    matrix = [[1,1,1,1], [0,0,1,1],[0,1,1,0],[0,1,1,0]]
    kernel = [[1,1],[0,1]]
    printMatrix(convnValid(matrix, kernel))

def testGetMaxIndex():
    print 'getMaxIndex'
    m = [13,3,1,6,10,11]
    print m
    print getMaxIndex(m)

if __name__ == '__main__':
    testScaleMatrix()
    testKronecker()
    testMatrixProduct()
    testCloneMatrix()
    testConvnValid()
    testGetMaxIndex()

