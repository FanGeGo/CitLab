#coding:utf-8
'''
个人引用行为，引文网络形成过程仿真器

完成仿真基本过程

'''
from domain import Domain


def simulate():
    ## 初始化一个领域
    d = Domain()
    d.go_to_year(8)

if __name__ == '__main__':
    simulate()





