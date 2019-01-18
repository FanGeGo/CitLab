#coding:utf-8
'''
定义需要用到的一些公用变量

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


## 研究周期分布函数，每个人的研究周期使用该函数进行随机抽样
rs_dis = json.loads(open('rs_dis.json').read())
rs_xs = [int(i) for i in rs_dis['x']]
rs_probs = [float(y) for y in rs_dis['y']]

## 生产力拟合函数
prod_dis = json.loads(open('prod_dis.json').read())
prod_xs = [int(i) for i in prod_dis['x']]
prod_probs = [float(y) for y in prod_dis['y']]
## 第一年进入的人没有0的抽样
prod_new_xs = prod_xs[1:]
prod_new_probs = np.array(prod_probs[1:])/np.sum(prod_probs[1:])

## 每一篇论文的价值系数lambda的抽样
lambda_dis = json.loads(open('data/lambda_dis.json').read())
lambda_list =lambda_dis['x']
lambda_probs = lambda_dis['y']

## 人数增减指数函数参数
AUTHOR_INCREASE_FUNC_PARAS = [4.4786305,0.05994124]


## 每篇论文平均参考文献数量
## N_REF = 30
def N_REF():
    return int(np.random.normal(30, 5, 1)[0])


def gen_id():
    return str(uuid.uuid1())

##根据现有人数，增加作者
def add_authors(t,s0,isfirst=False):
    ## 根据拟合指数增长参数[4.4786305  0.05994124]
    a = AUTHOR_INCREASE_FUNC_PARAS[0]
    b = AUTHOR_INCREASE_FUNC_PARAS[1]
    num_add = int(s0*a*np.exp(b*t))

    ## 初始的s0人需要加上
    if isfirst:
        num_add+=s0

    ## 每年进入的人上下浮动 0%~20%
    margin = int(np.random.normal(0,int(num_add*0.2),1)[0])
    num_add +=margin

    ### 每个人的声明周期进行模拟
    rses = np.random.choice(list(rs_xs),size=num_add,p=rs_probs,replace=True)

    # print num_add,np.max(rses),margin

    ## 根据离开概率，将进入的人进行
    return ['S_'+gen_id() for i in range(num_add)],rses

def ids_of_articles(num):
    return ['A'+gen_id() for i in range(num)]


## 作者写论文熟练的函数
def num_of_papers_to_write(age):

    ## 先设置为一个随机函数
    return random.randint(0,5)


def ID(mean):
    return np.random.poisson(mean)


## 价值增长函数, 先初始化为一个随机函数
def knowledge_gain_coef(num):
    return np.random.choice(lambda_list,size=num,p=lambda_probs,replace=True)

## 根据个人价值列表进行论文的引用
def cit_based_on_prob(articles,kgs,kg_probs):

    # print len(papers),len(kgs),len(kg_probs)

    ### 参考文献数量定义为30为均值，5为平均差的正态分布
    num_ref = N_REF()
    if len(articles)<=num_ref:
        return articles,kgs

    # print 'time:',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))

    ref_indexes = np.random.choice(range(len(articles)), size=num_ref, replace=False, p=kg_probs)

    # print 'time:',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))


    refs = []
    ref_kgs = []
    for i in ref_indexes:
        refs.append(articles[i])
        ref_kgs.append(kgs[i])

    return refs,ref_kgs

def read_papers(author,_author_rpapers,_ALL_article_id_set,_ALL_article_kg_dict,mode):

    ## 排除作者已读论文
    if len(_author_rpapers[author])>0:
        rpapers = set(_author_rpapers[author])
    else:
        rpapers = set([])
    unread_papers_set = _ALL_article_id_set-rpapers

    ## 随机阅读数量 3000为期望，500位均差
    np_read = int(np.random.normal(3000, 500, 1)[0])

    ## 如果论文总数量不足np_read,返回全部
    if len(unread_papers_set)<np_read:
        return unread_papers_set

    else:
        ## 几种假设
        # print 'time:',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        unread_papers_list = list(unread_papers_set)
        unread_paper_kgs = [_ALL_article_kg_dict[a] for a in unread_papers_list]
        # :
            # unread_papers_list.append(a)
            # unread_paper_kgs.append()

        # try:
        unread_paper_kg_probs = np.array(unread_paper_kgs)/float(np.sum(unread_paper_kgs))
        ## 随机抽取
        if mode=='random':
            nps = np.random.choice(range(len(unread_papers_list)),size=np_read,replace=False)

        elif mode=='top':
            ## 取价值最高的
            nps = sorted(range(len(unread_papers_list)),key=lambda x:unread_paper_kgs[x],reverse=True)[:np_read]

        elif mode=='prop':

            ## 按照价值概率随机选取
            nps = np.random.choice(range(len(unread_papers_list)),size=np_read,replace=False,p=unread_paper_kg_probs)

        # print 'time:',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))


        return [unread_papers_list[i]for i in nps]



def simulate_authors(length):
    ## 每年新人的数量
    year_news = defaultdict(set)
    ## 每年离开人的数量
    year_ends = defaultdict(set)

    ## 需要对数据进行存储
    ## 初始人数
    s0 = 10
    LENGTH = length
    ## 仿真120年
    print 'initail author num %d , simulation length %d ...' % (s0,LENGTH)

    print 'simulated authors ...'
    for i in range(1,LENGTH+1):
        print 'year %d ..'% i
        ## 根据模拟的函数进行增加人数，以及每个人对应的声明周期，最少一年
        authors,rses  = add_authors(i,s0,i==1)
        for j,a in enumerate(authors):

            ## 这个人的研究周期
            rs = rses[j]

            ## 开始年份为i
            start = i
            ## 结束年份为i+rs-1， 研究周期为1 代表当年结束
            end = start+rs-1

            year_news[start].add(a)
            year_ends[end].add(a)

    return year_news,year_ends


def simulate(length=100):

    year_news,year_ends =simulate_authors(length)
    # simulate_citation(year_news,year_ends,'ALL',length)
    simulate_citation(year_news,year_ends,'random',length)

    simulate_citation(year_news,year_ends,'top',length)

    simulate_citation(year_news,year_ends,'prop',length)

    simulate_citation(year_news,year_ends,'ALL',length)




### 对论文数量过程进行仿真
def simulate_citation(year_news,year_ends,mode='ALL',length=20):

    print 'MODE:',mode

    total_num_authors = 0
    ## 初始作者
    totals = set([])
    attrs = []
    ## 一位作者在某年发表的论文有哪些
    author_year_articles = defaultdict(lambda:defaultdict(list))

    ## 每一个元素是论文的属性,作者,年份,引用,价值
    article_list = []

    ## 全局论文库
    _ALL_articles_ids = []
    _ALL_kgs = []
    _ALL_kg_probs = []

    ## article_set
    _ALL_article_id_set = set([])
    _ALL_article_kg_dict = {}

    ## 个人论文库,记录作者阅读的论文id，kg
    _author_rpapers = defaultdict(list)

    print 'simulate wirting papers ...'
    ## 从第一年开始
    json_file = open('data/articles_jsons_{:}_{:}.txt'.format(mode,length),'w+')

    for i in sorted(year_news.keys()):

        print '------ In total, %d articles ...'%len(_ALL_articles_ids),'time:',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))

        _year_articles = []
        _year_kgs = []

        ## 该年的新人数量
        news = year_news[i]
        num_new = len(news)

        ## 写完论文后离开一部分人
        ends = year_ends[i]
        num_end = len(ends)



        au_states_list = []
        ## 对去年论文判断是否有人今年离开
        for pa in totals:
            ## 今年要离开的人
            if pa in ends:
                au_states_list.append([pa,-1])
            ## 其他人
            else:
                au_states_list.append([pa,1])

        for pa in news:
            au_states_list.append([pa,0])

        ## 对于au_state_list中的作者
        num_au = len(au_states_list)
        ### 对这些作者今年的论文进行随机抽样
        past_prods = np.random.choice(list(prod_xs),size=total_num_authors,p=prod_probs,replace=True)
        ## 对新的作者进行随机抽样
        new_prods = np.random.choice(list(prod_new_xs),size=num_new,p=prod_new_probs,replace=True)

        ## 剩余人数
        totals = totals|news
        totals = totals-ends
        total_num_authors=total_num_authors+num_new-num_end

        attrs.append([i,total_num_authors,num_new,num_end])
        print 'year %d, %d new authors, %d authors left, reserved total %d' % (i,num_new,num_end,total_num_authors)

        ##整体生产力分布
        prods = []
        prods.extend(past_prods)
        prods.extend(new_prods)

        for ia,(author,state) in enumerate(au_states_list):


            if ia%1000==0:
                print 'author progress 1:',ia,'/',num_au,'time:',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))

            ## 阅读论文,第一年没有论文可以读
            if mode!='ALL' and i>1:
                if ia%1000==0:
                    print 'read paper 1:', time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
                pns= read_papers(author,_author_rpapers,_ALL_article_id_set,_ALL_article_kg_dict,mode)
                _author_rpapers[author].extend(pns)
                if ia%1000==0:
                    print 'read paper 2:',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))

                ## 根据不同的模式对数据阅读论文
                # if mode=='random':
                #     _author_rpapers[author].extend(random_pns)
                # elif mode=='top':
                #     _author_rpapers[author].extend(top_pns)
                # elif mode=='prop':
                #     _author_rpapers[author].extend(prop_pns)

            ## 根据作者往年生产力平均水平进行随机项的添加
            num_of_papers =  random_pn(prods[ia],author_year_articles,author,state)

            ## 确定论文的ID
            article_ids = ids_of_articles(num_of_papers)

            ## 记录作者发表各年份发表的论文
            author_year_articles[author][i] = article_ids

            ##生成lambdas
            lambdas = knowledge_gain_coef(num_of_papers)

            if ia%1000==0:
                print 'author progress 2:',ia,'/',num_au,'time:',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))

            ## 根据全局引文库生成个人引文库
            if mode=='ALL':
                _personal_articles,_personal_kgs,_personal_kg_probs  = [_ALL_articles_ids,_ALL_kgs,_ALL_kg_probs]
            else:
                if i>1:
                    ## 个人库，以及个人库对应的论文价值
                    _personal_articles,_personal_kgs = _author_rpapers[author],[_ALL_article_kg_dict[aid] for aid in _author_rpapers[author]]
                    _personal_kg_probs = np.array(_personal_kgs)/float(np.sum(_personal_kgs))

            if ia%1000==0:
                print 'author progress 3:',ia,'/',num_au,'time:',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
            ## 对于每一篇论文来讲
            for aindex,aid in enumerate(article_ids):

                ## 如果是第一年
                if i==1:

                    ## 参考文献文献数量设置为[]
                    ref_list = []
                    ## 知识增益设置为100
                    kg= 100

                else:
                    ##在个人论文库中，根据kgs选择参考文献
                    # print np.sum(_personal_kg_probs)
                    ref_list,ref_kgs = cit_based_on_prob(_personal_articles,_personal_kgs,_personal_kg_probs)
                    kg = np.mean(ref_kgs)*lambdas[aindex]

                ## 存储该文章
                articleObj = {}
                articleObj['id'] = aid
                articleObj['author'] = author
                articleObj['kg'] = kg
                articleObj['refs'] = ref_list
                articleObj['year'] = i
                article_list.append(articleObj)

                _year_articles.append(aid)
                _year_kgs.append(kg)
                ## aid
                _ALL_article_id_set.add(aid)

                ## aid对应价值增益
                _ALL_article_kg_dict[aid] = kg

            if ia%1000==0:
                print 'author progress 4:',ia,'/',num_au,'time:',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))

        ## 论文的id以及论文的增益
        _ALL_articles_ids.extend(_year_articles)
        _ALL_kgs.extend(_year_kgs)
        ## 每年完成后，根据_ALL_kgs生成_ALL_kg_probs
        _ALL_kg_probs = np.array(_ALL_kgs)/float(np.sum(_ALL_kgs))

        ## 存储article_list中的论文
        lines = [json.dumps(a) for a in article_list]
        article_list = []


        print 'year %d, %d articles saved.'%(i,len(lines))
        json_file.write('\n'.join(lines)+'\n')

    json_file.close()
    print 'simulation done, %d articles are writen.'.format(len(_ALL_articles_ids))

    ## 完成仿真后，保存作者数据
    open('data/simulated_author_year_pn_{:}_{:}.json'.format(mode,length),'w').write(json.dumps(author_year_articles))
    print 'author year paper num dict saved to data/simulated_author_year_pn_{:}_{:}.json'.format(mode,length)

    ## 领域内作者变化曲线
    plt.figure(figsize=(5,4))
    years,totals,news,ends = zip(*attrs)

    plt.plot(years,news,label=u'新作者')
    plt.plot(years,ends,'--',label=u'离开作者')
    plt.plot(years,totals,label=u'剩余作者总数')

    plt.xlabel(u'年份',fontproperties='SimHei')
    plt.ylabel(u'作者数量',fontproperties='SimHei')
    plt.yscale('log')
    plt.legend(prop={'family':'SimHei','size':8})
    plt.tight_layout()

    plt.savefig('fig/simulated_author_num_{:}_{:}.png'.format(mode,length),dpi=400)
    print 'author num simulation saved to fig/simulated_author_num_{:}_{:}.png'.format(mode,length)


### 每一个作者生产力的随机项
def random_pn(prod,adict,a,state):

    mean = author_mean_pn(adict,a)

    ## 加上一个以mean为期望，以1位均差的正态随机项
    prod = prod+int(np.random.normal(mean,1,1)[0])

    if prod<0:
        prod=0

    ## 如果是作者今年离开，那么作者至少发表一篇论文
    if state==-1 and prod==0:
        prod=1

    return prod

def author_mean_pn(a_dict,a):

    year_dict = a_dict.get(a,{})

    values = year_dict.values()

    if len(values)==0:
        return 0
    else:
        return int(np.mean([len(v) for v in values]))




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
def validate_simulation(mode):
    ## 读数据

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
    for line in open('data/articles_jsons_{:}.txt'.format(mode)):

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
    plt.savefig('fig/simulated_life_length_dis_over_year_{:}.png'.format(mode),dpi=400)
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

    plt.savefig('fig/simulated_kg_dis_{:}.png'.format(mode),dpi=400)

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

    plt.title(u'仿真',fontproperties='SimHei')

    plt.xscale('log')
    plt.yscale('log')

    plt.tight_layout()

    plt.savefig('fig/simulated_citation_distribtuiotn_{:}.png'.format(mode),dpi=400)
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
    plt.savefig('fig/simulated_author_ref_dis_{:}.png'.format(mode),dpi=400)
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

    plt.savefig('fig/simulated_high_cited_paper_year_dis_{:}.png'.format(mode),dpi=400)

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
    plt.savefig('fig/simulated_high_life_length_dis_over_year_{:}.png'.format(mode),dpi=400)
    print 'fig saved to fig/simulated_high_life_length_dis_over_year.png'


if __name__ == '__main__':
    length = 100
    simulate(length=length)
    # simulate(mode='random',length=length)
    # simulate(mode='top',length=length)
    # simulate(mode='prop',length=length)


    # simulated_data_viz()
    # validate_simulation(mode='ALL')
    # validate_simulation(mode='random')

    # validate_simulation(mode='top')

    # validate_simulation(mode='prop')







