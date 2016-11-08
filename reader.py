#!usr/bin/python
#coding=utf-8
import random

class BitFileReader:
    def __init__(self):
        'pass'

    def open(self, filePath='./dataset/test-label'):
        self.filePath = filePath
        self.fileReader = open(self.filePath, 'rb')

    def close(self):
        self.filePath = ''
        self.fileReader.close()

    def read(self, returnType='un_byte'):
        if returnType == 'un_byte':
            content = self.fileReader.read(1)
            content = ord(content)
            #print content
            #print("%x"%(content)) #16进制格式打印
            return content
        elif returnType == 'int':
            # 其实这里转换成的貌似不是int，而是unsigned int
            content = self.fileReader.read(4)
            content = ord(content[0]) * 16777216 + ord(content[1]) * 65536 + ord(content[2]) * 256 + ord(content[3])
            #print content
            #print self.fileReader.tell()
            return content
    def step(self, number=1):
        self.fileReader.read(number);



'''
@description 读取文件bit
@param {string} filePath 文件路径
'''
def readFileBit(filePath='./foo'):
    numSize = 1
    with open(filePath, "rb") as f:
        byte = f.read(numSize)
        while byte != "":
            # print ord(byte)
            #print("%x"%(ord(byte)))
            byte = f.read(numSize)

#readFileBit()

'''
@description 读取文件
@param {string} tag 分割字符串
@param {string} filePath 文件路径
'''
def readFile(tag=' ', filePath='./foo'):
    f = open(filePath, 'rb') # 返回一个文件对象
    dataList = []
    line = f.readline() # 调用文件的 readline()方法
    while line:  
        line = line[0:-1]
        splitList = line.split(tag)
        dataList += splitList
        line = f.readline()
    f.close()
    return dataList

'''
@description 循环遍历二维列表vectors中的每一个一维列表
将其最后一项弹出，放入result中，直到有一个一维列表为空
@param {list} result 结果列表
@param {list[list]} vectors 二维列表
'''
def interInsert(result, vectors):
    tag = False
    while tag == False:
        for vList in vectors:
            if (len(vList) == 0):
                vectors.remove(vList)
                tag = True
                break
            result.append(vList.pop())

'''
@description 将二维列表vectors中的所有数据随机插入result中
@param {list} result 结果列表
@param {list[list]} vectors 二维列表
'''
def randInsert(result, vectors):
    emptyCount = 0
    resultSize = len(result)
    while emptyCount < len(vectors):
        emptyCount = 0
        for vList in vectors:
            if len(vList) == 0:
                emptyCount += 1
            else:
                index = random.randint(0, resultSize)
                result.insert(index, vList.pop())
                resultSize += 1

'''
@description 写入文件
@param {string} filePath 文件路径
@param {list} fileContent 内容
'''
def saveFile(filePath, fileContent):
    'pass'

def writeToFile(fileObject, data):
    if isinstance(data, list):
        for i in range(len(data)):
            writeToFile(fileObject, data[i])
    else:
        fileObject.write(str(data) + '\n')

def saveList(filePath, myList):
    f = open(filePath, 'w')
    writeToFile(f, myList)
    f.close()

def readFromList(contentList, dataSave, index):
    for i in range(len(dataSave)):
        if isinstance(dataSave[i], list):
            index = readFromList(contentList, dataSave[i], index)
        else:
            dataSave[i] = float(contentList[index])
            index += 1
    return index

'''
@description 从文件中读取list
@param {string} filePath 文件路径
@param {list} myList 需要存储的list
'''
def readListFromFile(filePath, myList):
    try:
        f = open(filePath, 'r')
    except:
        print '"%s" 文件不存在，读取失败' % filePath
        return False
    content = f.read()
    content = content.split('\n')
    readFromList(content, myList, 0)
    f.close()
    #print myList
    return True
'''
@description 逐行读文件，保存在列表中
@param {string} filePath 文件路径
@return {list}
'''
def readAsVecLine(filePath):
    'pass'
