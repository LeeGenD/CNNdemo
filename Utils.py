#!usr/bin/python
#coding=utf-8
import time
import reader

'''
返回列表
'''
def nList(i, j=0):
    if j==0:
        return [0] * i
    else:
        return [[0 for col in range(j)] for row in range(i)]

class TimeTest:
    def __init__(self):
        self.startTime = time.time()

    def getTimePass(self):
        return (time.time() - self.startTime)

'''
@description 打印图片
@param {list<list>} imageList 图片灰度map
'''
def printImage(imageList):
    printType = 1
    if printType == 1:
        # 打印二维列表
        for i in range(28):
            print imageList[i]
    elif printType == 2:
        # 打印二维列表
        for i in range(28):
            printStr = ''
            for j in range(28):
                value = imageList[i][j]
                if value < 85:
                    printStr += '  '
                elif value < 160:
                    printStr += '..'
                elif value < 245:
                    printStr += '::'
                else:
                    printStr += '**'
            print printStr
    elif printType == 3:
        # 打印一维列表
        for i in range(28):
            printStr = ''
            for j in range(28):
                value = imageList[i * 28 + j]
                if value < 85:
                    printStr += '  '
                elif value < 160:
                    printStr += '..'
                elif value < 245:
                    printStr += '::'
                else:
                    printStr += '**'
            print printStr
    elif printType == 4:
        # 打印一维列表
        for i in range(28):
            printStr = ''
            for j in range(28):
                value = imageList[i * 28 + j]
                if value:
                    printStr += '**'
                else:
                    printStr += ' '
            print printStr
    print ''

'''
@description 读取图片28*28的图片值
@param {BitFileReader} reader bit文件读取器
@returns {list<list>} 28*28的图片灰度map
'''
def readImage(reader):
    imageList = []
    for i in range(28):
        imageRowList = []
        for j in range(28):
            imageRowList.append(reader.read())
        imageList.append(imageRowList)
    return imageList

def transGrayToBinary(value):
    if value > 128:
        return 1
    else:
        return 0

def readImageLinear(reader):
    imageList = []
    for i in range(28):
        imageRowList = []
        for j in range(28):
            imageRowList.append(transGrayToBinary(reader.read()))
        imageList.append(imageRowList)
    return imageList
