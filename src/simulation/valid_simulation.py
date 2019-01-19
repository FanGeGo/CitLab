#coding:utf-8
'''
验证以及画图

'''
import uuid
import random
import numpy as np
import json
import sys
sys.path.extend(['.','..'])
from tools.basic_config import *
import powerlaw
import time


def simulated_data_viz(mode):

    author_year_articles_sim = json.loads(open('data/simulated_author_year_pn.json'.format(mode)).read())

    ## 所有作者的文章总数量
    tnas = []

    ## 领域内作者总数量
    year_an = defaultdict(int)

    ## 领域内文章总数量
    year_pn = defaultdict(list)

    ## 总数：年生产力
    tna_prod_dis = defaultdict(list)

    ## 对于每一位作者来讲
    for author in author_year_articles_sim.keys():

        total_num_of_articles = 0
        ## 每一年
        prod_list = []
        years = []
        for i,year in enumerate(sorted(author_year_articles_sim[author].keys(),key=lambda x:int(x))):

            ##第一年是作者进入的年
            if i==0:
                year_an[int(year)]+=1

            ## 文章数量
            num_of_articles = len(author_year_articles_sim[author][year])

            total_num_of_articles+=num_of_articles

            year_pn[int(year)].append(num_of_articles)

            prod_list.append(num_of_articles)

            years.append(int(year))


        tnas.append(total_num_of_articles)

        if years[-1]-years[0]>20:

            tna_prod_dis[total_num_of_articles].append(prod_list)

    ## 随着时间的增长领域内论文总数量
    xs = []
    ys = []
    an_ys = []
    total_pn = 0
    total_an = 0
    for year in sorted(year_pn.keys()):
        xs.append(year)
        total_pn+=np.sum(year_pn[year])

        total_an += year_an[year]
        an_ys.append(total_an)
        ys.append(total_pn)


    plt.figure(figsize=(5,4))

    plt.plot(xs,ys,label=u'文章总数')
    plt.plot(xs,an_ys,'--',label=u'作者总数')

    plt.xlabel(u'年份',fontproperties='SimHei')
    plt.ylabel(u'论文数量',fontproperties='SimHei')

    plt.title(u'仿真',fontproperties='SimHei')
    # plt.xscale('log')
    plt.yscale('log')
    plt.legend(prop={'family':'SimHei','size':8})

    plt.tight_layout()
    plt.savefig('fig/simulated_pn.png',dpi=400)
    print 'simulation of total papers saved to fig/simulated_pn.png'

    ## 画出作者的文章总数量分布
    tn_dict = Counter(tnas)
    xs = []
    ys = []
    for tn in sorted(tn_dict.keys()):

        xs.append(tn)
        ys.append(tn_dict[tn])

    print xs,ys
    plt.figure(figsize=(5,4))
    plt.plot(xs,ys,'o',fillstyle='none')
    plt.xlabel(u'作者文章总数量',fontproperties='SimHei')
    plt.ylabel(u'作者数',fontproperties='SimHei')
    plt.xscale('log')
    plt.yscale('log')
    plt.title(u'仿真',fontproperties='SimHei')

    plt.tight_layout()
    plt.savefig('fig/simulated_tn_dis.png',dpi=400)
    print 'data saved to fig/simulated_tn_dis.png'


    ### 6位最高产作者的可视化

    fig,axes= plt.subplots(2,3,figsize=(24,8))
    for i,tn in enumerate([40,60,80,90,95,100]):

        pn_list = tna_prod_dis[tn][0]

        ### 对每年生产力从高到低进行排序

        # sort_index = sorted(range(len(pn_list)),key=lambda x:pn_list[x],reverse=True)


        ax = axes[i/3,i%3]
        xs = range(1,len(pn_list)+1)
        ax.bar(xs,pn_list,label=u'总文章数=%d'%tn)

        # ax.set_yscale('log')
        # ax.set_xscale('log')

        ax.set_xticks(xs)
        ax.set_xticklabels(xs)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.set_xlim(-1,19)

        # ax.legend(loc=2)
        ax.legend(prop={'family':'SimHei','size':15},loc=2)


    plt.tight_layout()

    plt.savefig('fig/similated_tn_prod.png',dpi=400)

    print 'fig saved to fig/similated_tn_prod.png'


    fig,axes= plt.subplots(2,3,figsize=(24,8))
    for i,tn in enumerate([40,60,80,90,95,100]):

        pn_list = tna_prod_dis[tn][0]


        ### 对每年生产力从高到低进行排序

        sort_index = sorted(range(len(pn_list)),key=lambda x:pn_list[x],reverse=True)


        ax = axes[i/3,i%3]
        # xs = range(1,len(pn_list)+1)
        ax.bar(range(len(pn_list)),[pn_list[i] for i in sort_index],label=u'总文章数=%d'%tn)

        # ax.set_yscale('log')
        # ax.set_xscale('log')

        ax.set_xticks(range(len(pn_list)))
        ax.set_xticklabels(np.array(sort_index)+1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.set_xlim(-1,19)

        # ax.legend(loc=2)
        ax.legend(prop={'family':'SimHei','size':15})



    plt.tight_layout()

    plt.savefig('fig/simulated_tn_prod_sorted.png',dpi=400)

    print 'fig saved to fig/simulated_tn_prod_sorted.png'



### 根据生成的article，对仿真结果进行验证
def validate_simulation(mode,length):
    ## 读数据

    print 'MODE:',mode,',Length:',length

    ref_dict = defaultdict(int)
    ##
    ref_year_dict = defaultdict(lambda:defaultdict(int))

    progress = 0
    ## 论文发表年
    pid_year = {}
    ##kg的分布
    year_kgs = defaultdict(list)
    ##作者引用
    author_ref_num = defaultdict(lambda:defaultdict(int))
    ##作者数量
    author_paper_num = defaultdict(int)

    ## 价值增益的大小分布
    all_kgs = []
    for line in open('data/articles_jsons_{:}_{:}.txt'.format(mode,length)):
        progress+=1
        if progress%10000==0:
            print progress
        article = json.loads(line.strip())
        year = article.get('year',-1)
        pid = article['id']
        kg = article['kg']
        author_id = article['author']
        ref_list = article['refs']

        for ref in ref_list:

            ref_dict[ref]+=1
            author_ref_num[author_id][ref]+=1
            ref_year_dict[ref][year]+=1

        pid_year[pid] = year

        year_kgs[year].append(kg)

        author_paper_num[author_id]+=1

        all_kgs.append(kg)

    ## 生命周期分别
    year_lls = defaultdict(list)
    for ref in ref_year_dict.keys():

        year_dict = ref_year_dict[ref]

        years = year_dict.keys()

        lifelength = np.max(years)-np.min(years)

        year = pid_year[ref]

        year_lls[year].append(lifelength)

    ## 不同年终的平均长度
    xs = []
    ys = []
    for year in year_lls.keys():
        xs.append(year)
        ys.append(np.mean(year_lls[year]))

    plt.figure(figsize=(5,4))

    plt.plot(xs,ys)

    plt.xlabel(u'年份',fontproperties='SimHei')
    plt.ylabel(u'平均生命周期长度',fontproperties='SimHei')

    plt.tight_layout()
    plt.savefig('fig/simulated_life_length_dis_over_year_{:}_{:}.png'.format(mode,length),dpi=400)
    print 'fig saved to fig/simulated_life_length_dis_over_year.png'

    ## 价值增益分布

    _100_kgs = [kg for kg in all_kgs if kg>100]
    _200_kgs = [kg for kg in all_kgs if kg>200]

    print len(_100_kgs),len(_200_kgs),len(all_kgs)

    plt.figure(figsize=(5,4))

    plt.hist(all_kgs,bins=100,rwidth=0.5)
    plt.plot([100]*10,np.linspace(0,100000,10),'--',c='r',linewidth=2, label=u'初始价值增益')
    plt.xlabel(u'价值增益',fontproperties='SimHei')
    plt.ylabel(u'论文数量',fontproperties='SimHei')
    plt.yscale('log')
    plt.legend(prop={'family':'SimHei','size':8})

    plt.tight_layout()

    plt.savefig('fig/simulated_kg_dis_{:}_{:}.png'.format(mode,length),dpi=400)

    print 'kg dis saved to fig/simulated_kg_dis.png'

    ## 引文分布
    citation_nums = []

    high_cited_articles = []

    for ref in ref_dict.keys():

        cit_num = ref_dict[ref]

        if cit_num > 200:
            high_cited_articles.append(ref)

        citation_nums.append(cit_num)

    print '%d articles has citations.'%len(citation_nums)

    fit = powerlaw.Fit(citation_nums,xmin=1)

    print fit.power_law.xmin
    print 'compare:',fit.distribution_compare('power_law', 'exponential')

    num_dict = Counter(citation_nums)
    xs = []
    ys = []

    for num in sorted(num_dict):
        xs.append(num)
        ys.append(num_dict[num])

    ys = np.array(ys)/float(np.sum(ys))

    plt.figure(figsize=(5,4))

    plt.plot(xs,ys)

    plt.xlabel('$\#(c_i)$')
    plt.ylabel('$p(c_i)$')

    if mode =='ALL':
        tit = 'PCN-I'
    elif mode=='random':
        tit = 'PCN-II'
    elif mode == 'top':
        tit = 'PCN-III'
    else:
        tit = 'PCN-IV'

    plt.title(tit,fontproperties='SimHei')

    plt.xscale('log')
    plt.yscale('log')

    plt.tight_layout()

    plt.savefig('fig/simulated_citation_distribtuiotn_{:}_{:}.png'.format(mode,length),dpi=400)
    print 'citation distribution saved to simulated_citation_distribtuiotn.png'

    ## 作者引用分析
    ## 随机选择 20个作者，每个作者论文数量大于30
    author_candidates = [author for author in author_paper_num.keys() if author_paper_num[author]>30]

    authors = np.random.choice(author_candidates,size=20,replace=False)

    ig,axes = plt.subplots(4,5,figsize=(25,20))
    for ai,author in enumerate(authors):

        ref_dict = author_ref_num[author]


        refs = sorted(ref_dict.keys(),key=lambda x:ref_dict[x],reverse=True)

        xs = []
        ys = []

        for r,ref in enumerate(refs):
            xs.append(r+1)
            ys.append(ref_dict[ref])

        ax = axes[ai/5,ai%5]

        ax.plot(xs,ys,'o',label=u'论文数=%d'%author_paper_num[author])

        # ax.set_xscale('log')
        # ax.set_yscale('log')
        ax.legend(prop={'family':'SimHei','size':8})

        # ax.tight_layout()
    plt.tight_layout()
    plt.savefig('fig/simulated_author_ref_dis_{:}_{:}.png'.format(mode,length),dpi=400)
    print 'fig saved to fig/simulated_author_ref_dis.png'
    # return

    ### 高被引论文的分布
    years = []
    for pid in high_cited_articles:
        years.append(pid_year[pid])

    year_dis = Counter(years)
    xs = []
    ys = []
    for year in sorted(year_dis.keys()):
        xs.append(year)
        ys.append(year_dis[year])

    plt.figure(figsize=(5,4))

    plt.plot(xs,ys,label=u'高被引论文年份分布')

    plt.xlabel(u'年份',fontproperties='SimHei')
    plt.ylabel(u'论文数量',fontproperties='SimHei')
    plt.legend(prop={'family':'SimHei','size':8})
    plt.tight_layout()

    plt.savefig('fig/simulated_high_cited_paper_year_dis_{:}_{:}.png'.format(mode,length),dpi=400)

    print 'fig saved to fig/simulated_high_cited_paper_year_dis.png'

    ## 随机选择20个高被引论文进行可视化

    selected_highs=np.random.choice(high_cited_articles,size=20,replace=True)

    fig,axes = plt.subplots(4,5,figsize=(25,20))
    for hi,ref in enumerate(selected_highs):

        ax = axes[hi/5,hi%5]

        year_dict = ref_year_dict[ref]
        xs = []
        ys = []

        print 'year:%d, life:%d-%d' %(pid_year[ref],year_dict.keys()[0],year_dict.keys()[-1])
        for i,year in enumerate(sorted(year_dict.keys())):

            num = year_dict[year]

            xs.append(i)
            ys.append(num)

        ax.plot(xs,ys)
        ax.set_xlabel(u'年份',fontproperties='SimHei')
        ax.set_ylabel(u'引用次数',fontproperties='SimHei')
        ax.set_title(u'total')

    plt.tight_layout()
    plt.savefig('fig/simulated_high_life_length_dis_over_year_{:}_{:}.png'.format(mode,length),dpi=400)
    print 'fig saved to fig/simulated_high_life_length_dis_over_year.png'

if __name__ == '__main__':
    # simulated_data_viz()
    # validate_simulation('ALL',length)
    # validate_simulation('random',length)

    # validate_simulation('top',length)

    # validate_simulation('prop',length)

    # validate_simulation(mode='random')

    # validate_simulation(mode='top')

    # validate_simulation(mode='prop')
