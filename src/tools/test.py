#coding:utf-8
'''

测试

'''
from basic_config import *

def plot_func(x):

    print 2.5+x**-1.5
    print 2.5+(x-1)**-1.5
    y = (2.5+x**-1.5)/(2.5+(x-1)**-1.5)
    print 'y:',y
    return y



def plot_f():

    xs = range(2,15)

    ys = [plot_func(x) for x in xs]

    print xs
    print ys


    plt.figure()

    plt.plot(xs,ys)

    plt.savefig('test.png')


if __name__ == '__main__':
    plot_f()