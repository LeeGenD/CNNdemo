#!usr/bin/python
#coding=utf-8
import Calc
import random
'''
返回列表
'''
def createList(i, j=0, k=0, t=0):
    if j==0:
        return [getRandomNumber()] * i
    elif k==0:
        return [[getRandomNumber() for col in range(j)] for row in range(i)]
    elif t==0:
        return [[[getRandomNumber() for kol in range(k)] for col in range(j)] for row in range(i)]
    else:
        return [[[[getRandomNumber() for tol in range(t)] for kol in range(k)] for col in range(j)] for row in range(i)]

def getSize(i, j):
    return i - j + 1

def getRandomNumber():
    return (random.randint(0, 1024) - 512) / 512.0

def colorPrint(str):
    print '\033[1;32;40m' + str + '\033[0m'


THRESHOD = 0.99 #正确率
RATE =0.1 #学习速率
TRAIN_TIMES = 2 #最大训练次数

TRAIN_FILE = './dataset/train-images-idx3-ubyte' #训练文件
LABEL_FILE = './dataset/train-labels-idx1-ubyte'
TEST_FILE = './dataset/train-labels-idx1-ubyte' #测试文件

KERNEL_I_C1_FILE = './save/kernel_i_c1'
KERNEL_S2_C3_FILE = './save/kernel_s2_c3'
KERNEL_S4_O_FILE = './save/kernel_s4_o'
BIAS_I_C1 = './save/bias_i_c1'
BIAS_S2_C3 = './save/bias_s2_c3'
BIAS_S4_O = './save/bias_s4_o'

'''
偏移量和权值矩阵需要训练，作为全局变量
I: input 输入
C: convolution 卷积
S: sample 采样
O: output 输出
'''
I_SIZE = 28

C1_SIZE = 24
C1_NUM = 6

S2_SIZE = 12

C3_SIZE = 8
C3_NUM = 12

S4_SIZE = 4

O_NUM = 10

kernel_i_c1 = createList(C1_NUM, 1, 5, 5)
bias_i_c1 = createList(C1_NUM)


kernel_s2_c3 = createList(C3_NUM, C1_NUM, 5, 5)
bias_s2_c3 = createList(C3_NUM)


kernel_s4_o = createList(O_NUM, C3_NUM, 4, 4)
bias_s4_o = createList(O_NUM)

if __name__ == '__main__':
    colorPrint('kernel_i_c1:')
    for i in range(len(kernel_i_c1)):
        Calc.printMatrix(kernel_i_c1[i])

    colorPrint('kernel_s2_c3:')
    for i in range(len(kernel_s2_c3)):
        Calc.printMatrix(kernel_s2_c3[i])

