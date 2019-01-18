#coding:utf-8
'''
定义常用的函数

'''
import numpy as np

## 指数
def exp_func(t,a,b):

    return 9*a*np.exp(b*t)

def powlaw(t,a,b):

    return a*t**(-b)

