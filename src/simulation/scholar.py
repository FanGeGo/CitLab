#coding:utf-8
'''

学者仿真器

'''
import sys
sys.path.extend(['.','..'])
from tools.basic_config import *
import commons



class Scholar:

    def __init__(self,age,viewed_papers):

        ## scholar的学术年龄
        self._age = age

        ## 该学者发表的论文数量
        self._total_num_of_paper = 0

        ## 看过的论文集合
        self._viewed_papers = viewed_papers


    @property
    def total_number_of_papers(self):
        return self._total_num_of_paper

    @property
    def scholar_age(self):
        return self._age


    ##撰写一篇论文，从看过的论文中进行选择引文，引文选择的概率是cit_pro_func
    def write_one_paper(self):
        ## 从看到过的论文中进行按照概率随机选择
        papers,years,kgs,kg_probs = self._viewed_papers
        #### 根据所看到的论文进行概率计算
        refs,ref_kgs = commons.cit_based_on_prob(papers,kgs,kg_probs)
        ## 根据参考文献值进行随机增益
        kg = commons.knowledge_gain(ref_kgs)
        ## 给出一个随机ID
        ID = 'P_'+commons.gen_id()
        return ID,refs,kg

    ## 写一年的论文
    def write_papers(self):
        ## 得到今年该作者随机的写作数量
        num_in_this_year = commons.num_of_papers_to_write(self._age)
        ## 根据写作数量写文章
        attrs = []
        for i in range(num_in_this_year):
            _id,ref_list,kg = self.write_one_paper()
            attrs.append([_id,ref_list,kg])

        return attrs

