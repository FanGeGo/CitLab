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

## 作者从未读的论文中按照mode进行阅读论文
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

### 仿真作者数量变化
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

### 仿真作者写论文数量变化
def simulate_author_papers(year_news,year_ends,length):

    total_num_authors = 0
    ## 初始作者
    totals = set([])
    attrs = []
    ## 一位作者在某年发表的论文有哪些
    author_year_articles = defaultdict(lambda:defaultdict(list))

    ##
    _ALL_articles_ids = []

    print 'simulate wirting papers ...'
    ## 从第一年开始
    for i in sorted(year_news.keys()):

        print '------ In total, %d articles ...'%len(_ALL_articles_ids),'time:',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))

        _year_articles = []

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

            ## 根据作者往年生产力平均水平进行随机项的添加
            num_of_papers =  random_pn(prods[ia],author_year_articles,author,state)
            ## 确定论文的ID
            article_ids = ids_of_articles(num_of_papers)
            ## article id
            author_year_articles[author][i] = article_ids

            _year_articles.extend(article_ids)


        _ALL_articles_ids.extend(_year_articles)


    print 'total authors:',len(author_year_articles.keys()),'; number of articles:',len(_ALL_articles_ids)
    ## 保存作者年论文的json
    open('data/simulate_author_year_papers_{:}.json'.format(length),'w').write(json.dumps(author_year_articles))

    print 'author simulate data saved to data/simulate_author_year_papers_{:}.json'.format(length)

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

    plt.savefig('fig/simulated_author_num_{:}.png'.format(length),dpi=400)
    print 'author num simulation saved to fig/simulated_author_num_{:}.png'.format(length)

    return author_year_articles

### 对论文数量过程进行仿真按照不同的模式进行引用
def simulate_citation(year_news,year_ends,author_year_articles,mode='ALL',length=20):

    print 'MODE:',mode
    ## 每一个元素是论文的属性,作者,年份,引用,价值
    article_list = []

    ## 所有作者集合
    totals = set([])
    total_num_authors = 0

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

        ## 剩余人数
        totals = totals|news
        totals = totals-ends
        total_num_authors=total_num_authors+num_new-num_end

        print 'year %d, %d new authors, %d authors left, reserved total %d' % (i,num_new,num_end,total_num_authors)

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

            article_ids = author_year_articles[author][i]
            num_of_papers = len(article_ids)
            ##生成lambdas
            lambdas = knowledge_gain_coef(num_of_papers)

            ## 根据全局引文库生成个人引文库
            if mode=='ALL':
                _personal_articles,_personal_kgs,_personal_kg_probs  = [_ALL_articles_ids,_ALL_kgs,_ALL_kg_probs]
            else:
                if i>1:
                    ## 个人库，以及个人库对应的论文价值
                    _personal_articles,_personal_kgs = _author_rpapers[author],[_ALL_article_kg_dict[aid] for aid in _author_rpapers[author]]
                    _personal_kg_probs = np.array(_personal_kgs)/float(np.sum(_personal_kgs))

            if ia%1000==0:
                print 'author progress 2:',ia,'/',num_au,'time:',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
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
                print 'author progress 3:',ia,'/',num_au,'time:',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))

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



def simulate(length=100):
    ##仿真的每年的作者数量
    year_news,year_ends =simulate_authors(length)

    ## 仿真作者的文章数量也需要一致
    author_year_articles = simulate_author_papers(year_news,year_ends,length)

    simulate_citation(year_news,year_ends,author_year_articles,'random',length)

    simulate_citation(year_news,year_ends,author_year_articles,'top',length)

    simulate_citation(year_news,year_ends,author_year_articles,'prop',length)

    simulate_citation(year_news,year_ends,author_year_articles,'ALL',length)



if __name__ == '__main__':
    length = 120
    simulate(length=length)
    # simulate(mode='random',length=length)
    # simulate(mode='top',length=length)
    # simulate(mode='prop',length=length)









