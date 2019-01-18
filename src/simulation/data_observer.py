#coding:utf-8
'''

    使用APS的数据进行学者数量、文章数量的数据观察以及规律验证。

    1. 按照 http://science.sciencemag.org/content/sci/suppl/2016/11/02/354.6312.aaf5239.DC1/Sinatra.SM.pdf 中的作者名消歧方法进行作者消歧
        首先名字全匹配，然后匹配机构，如果没有机构就舍弃。

    2. 算上合著的情况： 文章作者数量分布，作者生产力


'''
from scipy.misc import factorial
import sys
sys.path.extend(['.','..'])
from tools.basic_config import *

# from funcs import exp_func
from funcs import powlaw
import powerlaw


mpl.rcParams['axes.unicode_minus'] = False

DATA_PATH = '/Users/huangyong/Workspaces/Study/datasets/APS/aps-dataset-metadata-2016'
AUTHOR_JSON_PATH = 'data/author_year_articles.json'
NUM_AUTHOR_DIS_PATH = 'data/author_num_dis.json'

def list_metadata():
    for journal in os.listdir(DATA_PATH):
        for issue in os.listdir(DATA_PATH+'/'+journal):

            for article in os.listdir(DATA_PATH+'/'+journal+'/'+issue):

                if not article.endswith('json'):
                    continue

                yield DATA_PATH+'/'+journal+'/'+issue+'/'+article


def extract_author_info(article_json):
    pid = article_json['id']
    authors = article_json.get('authors','-1')
    date = article_json.get('date','-1')
    atype = article_json.get('articleType','-1')
    affiliations = article_json.get('affiliations',[])

    return pid,authors,date,atype,affiliations


def extract_from_metadata():
    max_year= 0
    min_year= 3000
    author_year_articles = defaultdict(lambda:defaultdict(list))
    authornum_dis = defaultdict(int)
    progress = 0
    empty_authors = 0
    for article_path in list_metadata():
        article_json = json.loads(open(article_path).read())
        pid,authors,date,atype,affiliations = extract_author_info(article_json)
        afid_name = {}
        for affiliation in affiliations:
            afid = affiliation['id']
            af_name = affiliation['name']
            afid_name[afid] = af_name

        ## 只计算存在作者信息、日期信息的article
        if authors=='-1' or len(authors)==0 or date =='-1' or atype!='article' or len(authors)>10:
            continue

        year = int(date.split('-')[0])

        if year > max_year:
            max_year = year

        if year < min_year:
            min_year = year

        progress+=1
        if progress%10000==0:
            print 'progress %d ....' % progress
            # break

        num_of_author = len(authors)
        authornum_dis[num_of_author] += 1

        ## 在这里首先只算第一作者

        isNull = 0
        for author in authors:
            # print author, affiliations,afid_name
            name = author['name']

            affilids = author.get('affiliationIds',[])
            # affiliations = author.get('affiliations',[])

            ## 不存在机构数据置空
            if len(affilids)==0:
                continue

            aff_names = []
            for afid in sorted(affilids):
                aff = afid_name.get(afid,'-1')
                if aff==-1:
                    continue
                aff_names.append('_'.join(aff.replace('.','').replace(',','').lower().split()))

            if len(aff_names)==0:
                continue

            isNull+=1
            name_aff = name+'_'+'_'.join(aff_names).lower()

            # print name_aff

            author_year_articles[name_aff][year].append(pid)

        if isNull ==0:
            empty_authors+=1

    print 'empty authors %d' % empty_authors

    print '%d aritcles processed, %d authors reserved ...' % (progress, len(author_year_articles.keys()))
    print 'from %d to %d years ...' % (min_year, max_year)

    open(AUTHOR_JSON_PATH,'w').write(json.dumps(author_year_articles))

    print 'data saved to %s.' % AUTHOR_JSON_PATH

    open(NUM_AUTHOR_DIS_PATH,'w').write(json.dumps(authornum_dis))

    print 'author num dis saved to %s' % NUM_AUTHOR_DIS_PATH


def author_productivity():

    author_year_articles = json.loads(open(AUTHOR_JSON_PATH).read())

    ## year 不同年分钟作者的平均生产力
    year_products = defaultdict(list)
    year_articles = defaultdict(list)

    ## 每个研究年龄对应的最大的生产力

    age_papernum = defaultdict(list)

    avg_pns = defaultdict(list)

    papernum_dis = defaultdict(int)
    paper_nums = []
    for author in author_year_articles.keys():

        years  = [int(y) for y in author_year_articles[author].keys()]


        max_year = np.max(years)
        min_year = np.min(years)

        if max_year-min_year <19:
            continue

        tn = 0
        pns = []
        for i,year in enumerate(range(min_year,max_year+1)):

            year  = str(year)

            articles = author_year_articles[author].get(year,[])

            year_articles[int(year)].extend(articles)

            paper_num = len(articles)
            paper_nums.append(paper_num)

            tn+=paper_num

            papernum_dis[paper_num]+=1

            pns.append(paper_num)

            # print i,paper_num

            age_papernum[i+1].append(paper_num)

            year_products[int(year)].append(paper_num)

        avg_pn = np.mean(pns)

        avg_pns[tn].append(pns)


    #papernum_dis

    xs = []
    ys = []

    for pn in sorted(papernum_dis.keys()):
        xs.append(pn)
        ys.append(papernum_dis[pn])

    plt.figure(figsize=(5,4))

    ys = np.array(ys)/float(np.sum(ys))

    expfunc = lambda t,a,b:a*np.exp(b*t)
    popt,pcov = scipy.optimize.curve_fit(expfunc,xs[3:],ys[3:],p0=(0.2,-2))

    print popt

    plt.plot(np.array(xs),ys,label=u'作者年生产力')
    plt.plot(xs,[expfunc(x,*popt) for x in xs],'--',label=u'拟合曲线$p(n)=%.2f*e^{%.2fn}$'%(popt[0],popt[1]))

    fit=powerlaw.Fit(np.array(paper_nums)+1,discrete=True,xmin=2)

    print 'xmin:',fit.power_law.xmin
    print 'alpha:',fit.power_law.alpha
    print 'sigma:',fit.power_law.sigma

    print 'compare:',fit.distribution_compare('power_law', 'exponential')
    # print 'compare:',fit.distribution_compare('power_law', 'exponential')

    # plaw = lambda t,a,b: a*t**(-fit.power_law.alpha)*np.exp(b*t)
    # popt,pcov = scipy.optimize.curve_fit(plaw,np.array(xs)+1,ys,p0=(0.2,-1))

    # plt.plot(xs,[plaw(x+1,*popt) for x in xs],'-^')



    plt.xlabel(u'作者单年生产力',fontproperties='SimHei')
    plt.ylabel(u'概率',fontproperties='SimHei')

    plt.yscale('log')
    # plt.xscale('log')
    plt.legend(prop={'family':'SimHei','size':8})

    plt.tight_layout()

    plt.savefig('fig/prod_dis.png',dpi=400)

    print 'fig saved to fig/prod_dis.png'

    ### 保存一个作者的生产力随机抽样

    xs = list(range(0,41))

    ys = [float(expfunc(x+1,*popt)) for x in xs]

    ys = list(np.array(ys)/np.sum(ys))


    prod_dis = {}

    prod_dis['x'] = xs
    prod_dis['y'] = ys

    open('prod_dis.json','w').write(json.dumps(prod_dis))
    print 'author productivity distribution saved to prod_dis.json'
    # return

    ## 选取的6个

    tns =  sorted(avg_pns.keys())[-6:]

    fig,axes= plt.subplots(2,3,figsize=(24,8))
    for i,tn in enumerate(tns):

        pn_list = avg_pns[tn][0]

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

    plt.savefig('fig/tn_prod.png',dpi=400)

    print 'fig saved to fig/tn_prod.png'

    ### 排序

    tns =  sorted(avg_pns.keys())[-6:]

    fig,axes= plt.subplots(2,3,figsize=(24,8))
    for i,tn in enumerate(tns):

        pn_list = avg_pns[tn][0]

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

    plt.savefig('fig/tn_prod_sorted.png',dpi=400)

    print 'fig saved to fig/tn_prod_sorted.png'

    return

    ## 选择


    # expfunc = lambda t,a,b:a*np.exp(b*t)
    ## 对所有年龄进行拟合

    # color = plt.cm.viridis(np.linspace(0.01,0.99,7)) # This returns RGBA; convert:
    # hexcolors = map(lambda rgb:'#%02x%02x%02x' % (rgb[0]*255,rgb[1]*255,rgb[2]*255),
    #            tuple(color[:,0:-1]))

    # plt.figure(figsize=(5,4))
    # ages = []
    # ks = []
    # for age in range(1,20):

    #     pn_list = age_papernum[age]

    #     pn_dis = Counter(pn_list)

    #     xs = []
    #     ys = []
    #     for pn in range(20):

    #         n = pn_dis.get(pn,0)

    #         xs.append(pn)
    #         ys.append(n)

    #     ys = np.array(ys)/float(np.sum(ys))
    #     ## 使用指数分布进行拟合
    #     popt,pcov = scipy.optimize.curve_fit(expfunc,xs,ys,p0=(0.2,-1))

    #     ages.append(age)
    #     ks.append(popt[1])

    #     ## 画出来
    #     print age,popt
    #     if age in [1,2,5,10,15,20]:
    #         plt.plot(xs,[expfunc(x,*popt) for x in xs],label='$k=%d$'%age,c=hexcolors[[1,2,5,10,15,20].index(age)])

    # plt.xlim(-1,20)
    # plt.xlabel(u'论文数量',fontproperties='SimHei')
    # plt.ylabel(u'概率',fontproperties='SimHei')
    # plt.yscale('log')
    # plt.xticks(range(20),range(20))
    # plt.legend()
    # plt.tight_layout()

    # plt.savefig('fig/pn_exponential.png',dpi=400)
    # print 'fig saved to fig/pn_exponential.png'


    # plt.figure(figsize=(5,4))

    # plt.plot(ages,ks)

    # plt.xlabel(u'研究年龄',fontproperties='SimHei')
    # plt.ylabel(u'$\lambda$')

    # plt.tight_layout()

    # plt.savefig('fig/para_dis.png',dpi=400)

    # print 'fig saved to fig/para_dis.png'

    # return


    ## 抽样查看分布
    fig,axes= plt.subplots(2,3,figsize=(18,6))
    for i,age in enumerate([1,2,5,10,15,20]):

        pn_list = age_papernum[age]

        pn_dis = Counter(pn_list)

        xs = []
        ys = []
        for pn in range(20):

            n = pn_dis.get(pn,0)

            xs.append(pn)
            ys.append(n)

        ys = np.array(ys)/float(np.sum(ys))

        # print ys

        # fit=powerlaw.Fit(np.array(pn_list)+1,discrete=True,xmin=1)
        # print fit.power_law.alpha
        ## 使用指数分布进行拟合
        # popt,pcov = scipy.optimize.curve_fit(expfunc,xs,ys)

        # print age,popt

        ax = axes[i/3,i%3]

        ax.bar(xs,ys,label='$k=%d$' % age)

        ax.set_yscale('log')
        # ax.set_xscale('log')

        ax.set_xticks(xs)
        ax.set_xticklabels(xs)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(-1,19)

        ax.legend()


    plt.tight_layout()

    plt.savefig('fig/age_prod.png',dpi=400)

    print 'fig saved to fig/age_prod.png'

    # return

    ## 每年的平均生产力是每个作者的生产力的平均值
    xs = []
    ys = []
    ppas = []

    for year in sorted(year_products.keys()):

        xs.append(year)
        avg_products = np.mean(year_products[year])
        paper_num = len(set(year_articles[year]))
        author_num = len(year_products[year])

        ppas.append(paper_num/float(author_num))
        ys.append(avg_products)

    plt.figure(figsize=(5,4))
    plt.plot(xs,ys,label=u'平均生产力')
    plt.plot(xs,ppas,'--',label=u'平均文章数')
    plt.xlabel(u'年份',fontproperties='SimHei')
    plt.ylabel(u'生产力',fontproperties='SimHei')
    plt.ylim(0,2)
    plt.legend(prop={'family':'SimHei','size':8})
    plt.tight_layout()

    plt.savefig('fig/productivity.png',dpi=400)

    print 'fig saved to fig/productivity.png'

    ## 随着年龄的增加，最大生产力
    xs = []
    ys = []
    means = []
    nums = []
    for age in range(1,21):
        xs.append(age)
        sort_nums = sorted(age_papernum[age],reverse=True)
        ## 最高5%的数的平均值
        tn = int(len(sort_nums)*0.01)+1
        nums.append(len(sort_nums))
        max_papernum = np.mean(sort_nums[:tn])

        means.append(np.mean(sort_nums))
        ys.append(max_papernum)

    # def poisson(k, lamb,b):
    #     return b*(lamb**k/factorial(k)) * np.exp(-lamb)

    # plt.figure(figsize=(5,4))

    # plt.plot(xs,nums)
    # print nums
    # plt.xlabel(u'研究年龄',fontproperties='SimHei')
    # plt.ylabel(u'作者数量',fontproperties='SimHei')

    # plt.tight_layout()

    # plt.savefig('fig/age_num_dis.png',dpi=400)


    # def multi(x,a,b,c):
    #     return a*x*x+b*x+c

    # parameters, cov_matrix = scipy.optimize.curve_fit(multi, xs, ys)
    # print parameters

    # fit_Y = [multi(x,*parameters) for x in xs]

    plt.figure(figsize=(5,4))
    plt.plot(xs,ys,'-o',label='最大生产力')
    plt.plot(xs,means,'-^',label='平均生产力')
    # plt.plot(xs,means)
    plt.xlabel(u'研究年龄',fontproperties='SimHei')
    plt.ylabel(u'最大生产力',fontproperties='SimHei')

    plt.xticks(xs,xs)

    # plt.plot(xs,fit_Y,label=u'拟合曲线')
    plt.legend(prop={'family':'SimHei','size':8})

    plt.ylim(0,8)

    plt.tight_layout()

    plt.savefig('fig/max_producitvity.png',dpi=400)

    print 'fig saved to fig/max_producitvity.png'


def delta_num():
    ## 根据author的信息计算，每年新增的学者的数量
    year_new_num = defaultdict(int)
    ## 每年不再继续研究的学者的数量
    year_end_num = defaultdict(int)
    ## 研究年龄分布比例就是年龄离开的概率
    lifespan_dis = defaultdict(int)
    ## 研究寿命与总文献数量
    lifespan_nums = defaultdict(list)
    ## 年龄与下一年的数量
    age_num = defaultdict(list)
    author_year_articles = json.loads(open(AUTHOR_JSON_PATH).read())
    ## 文章数量的
    article_num_dis = defaultdict(int)

    intervals = []

    used_au_num = 0
    out_interval = 0

    year_lifespan = defaultdict(list)

    for author in author_year_articles.keys():

        paper_years =  sorted(author_year_articles[author].keys(),key=lambda x:int(x))

        last_year = 0
        max_interval = 0
        for i,y in enumerate(paper_years):
            y = int(y)
            if i>0:
                interval = y - last_year
                # if interval >1:

                if interval > max_interval:
                    max_interval = interval

            last_year = y

        if max_interval>0:
            intervals.append(max_interval)

        if max_interval>10:
            out_interval+=1
            continue

        used_au_num +=1

        # print paper_years
        article_num = 0
        for i,year in enumerate(paper_years):
            anum=len(author_year_articles[author][year])
            article_num +=anum
            age_num[i].append(anum)

        lifespan = int(paper_years[-1])-int(paper_years[0])+1

        year_lifespan[int(paper_years[0])].append(lifespan)

        lifespan_dis[lifespan]+=1

        lifespan_nums[lifespan].append(article_num)

        article_num_dis[article_num]+=1

        startYear = int(paper_years[0])
        endYear = int(paper_years[-1])

        year_new_num[startYear]+=1
        year_end_num[endYear]+=1

    ## 每年各个比例
    left_percent = defaultdict(list)
    years = []

    ## 统计1980-2006之间的均值
    _year_left = defaultdict(list)
    _year_percents = defaultdict(list)
    for year in sorted(year_lifespan.keys()):
        if year > 2006:
            continue
        ta = float(len(year_lifespan[year]))
        lifespans = Counter(year_lifespan[year])
        years.append(year)
        for lifespan in range(1,11):
            percent = lifespans.get(lifespan,0)/ta

            left_percent[lifespan].append(percent)

        if year>1980 and year <= 2006:
            for lifespan in sorted(lifespans.keys()):
                _year_left[year].append(lifespan)
                _year_percents[year].append(lifespans[lifespan])

    fig,ax1 = plt.subplots(figsize=(5,4))

    ## 计算均值
    _left_avgs = defaultdict(list)
    for year in sorted(_year_left.keys()):

        _left = _year_left[year]
        _percents = _year_percents[year]

        for i,l in enumerate(_left):

            _left_avgs[l].append(_percents[i])


        ax1.plot(_left,np.array(_percents)/float(np.sum(_percents)),'^',mec='#D2D2D2',mew=0.5)

    avg_xs = []
    avg_ys = []

    for _l in sorted(_left_avgs.keys()):
        avg_xs.append(_l)
        avg_ys.append(np.mean(_left_avgs[_l]))

    ax1.plot(avg_xs,np.array(avg_ys)/np.sum(avg_ys),'-.',linewidth=3,label=u'均值',c='r')

    # ax1.plot([10]*10,np.linspace(0.0001,1,10),'--',linewidth=0.5)

    ax1.set_xlabel(u'研究周期(年)',fontproperties='SimHei')
    ax1.set_ylabel(u'$p_{rs}(t)$')
    ax1.set_yscale('log')
    # ax1.set_xscale('log')

    # _2000_cd = []
    # _2000_cont = []
    # _2000_ta = float(np.sum(_2000_percents))
    # _2000_tp = 0
    # last_cont = 1
    # for i,p in enumerate(_2000_percents):
    #     _2000_tp+=p
    #     _2000_cd.append(_2000_tp/_2000_ta)
    #     cont = (1-_2000_tp/_2000_ta)/last_cont
    #     _2000_cont.append(cont)
    #     last_cont = cont

    # ###P_c先不画
    # print _2000_cont
    # # l3 = ax1.plot(_2000_left,_2000_cont,'-s',label='$P_c(s_i,t)$',mec='#D2D2D2',mew=0.5)

    # ax2 = ax1.twinx()
    # l2 = ax2.plot(_2000_left, _2000_cd,'-o',c='r',label='离开概率',mec='#D2D2D2',mew=0.5)
    # ax2.set_ylabel('$P_l(t)$', color='r')
    # ax2.tick_params('y', colors='r')
    # # lns = l1+l3+l2
    # lns = l1+l2

    # labels = [l.get_label() for l in lns]
    plt.legend(prop={'family':'SimHei','size':8},loc=8)


    plt.tight_layout()

    plt.savefig('fig/_avg_left_dis.png',dpi=400)

    print 'year 2000 savd to fig/_avg_left_dis.png'

    ### 对密度概率分布进行拟合
    fig,ax = plt.subplots(figsize=(5,4))
    _ALL_SPANS = []

    for y in range(1980,2000):
        _ALL_SPANS.extend([l for l in year_lifespan[y]])
    xs = avg_xs
    ys = np.array(avg_ys)/float(np.sum(avg_ys))
    print xs
    print ys
    l1=ax.plot(xs,ys,'-o',label='研究周期',mec='#D2D2D2',mew=0.5)
    fit=powerlaw.Fit(_ALL_SPANS,discrete=True,xmin=2)
    alpha = fit.power_law.alpha
    print 'xmin\t=',fit.xmin
    print 'alpha\t=',fit.power_law.alpha
    print 'sigma\t=',fit.power_law.sigma
    # fit.plot_pdf(ax=ax)
    # fit.power_law.plot_pdf(linestyle='--',c='r',label=u'拟合曲线',ax=ax)
    ax.set_xlabel(u'研究周期',fontproperties='SimHei')
    ax.set_ylabel(u'$p_{rs}(t)$')
    # ax.set_xscale('log')
    ax.set_yscale('log')

    pl_func = lambda t,a:a*t**(-alpha)

    # popt, pcov = scipy.optimize.curve_fit(powlaw,xs[:10],ys[:10],p0=(0.8,2.0))
    # fit_Y =  [powlaw(x,*popt) for x in xs]
    # l2=ax.plot(xs[:],fit_Y[:],'--',label=u'拟合曲线 $p_{rs}(t) = %.2f*t^{-%.2f}$ \n$t_{min}=3$' % (popt[0],popt[1]))

    # print popt

    popt, pcov = scipy.optimize.curve_fit(pl_func,xs[1:],ys[1:],p0=(0.8))
    fit_Y =  [pl_func(x,*popt) for x in xs]

    print popt
    l2=ax.plot(xs[1:],fit_Y[1:],'--',label=u'拟合曲线 $p_{rs}(t) = %.2f*t^{-%.2f}$' % (popt[0],alpha))

    # def poisson(k, lamb):
        # return (lamb**k/factorial(k)) * np.exp(-lamb)

    # parameters, cov_matrix = scipy.optimize.curve_fit(poisson, xs, ys[:10])

    # l3=ax.plot(xs,poisson(xs,*parameters),'--',label=u'泊松分布')


    ##保存一张图片对研究周期分布的拟合

    # lns = l1+l2
    # labels = [l.get_label() for l in lns]
    plt.legend(prop={'family':'SimHei','size':8})
    plt.tight_layout()
    plt.savefig('fig/left_fit.png',dpi=400)

    print 'left prop saved to fig/left_fit.png.'

    ### 以拟合曲线进行离开概率的模拟
    ## 模拟四十年的周期, 这里存在后面的和大于1的问题，目前只模拟到10年
    fit_X = range(1,41)
    fit_Y =  [pl_func(x,*popt) for x in fit_X]
    fig,ax1 = plt.subplots(figsize=(5,4))
    real_Y = []
    real_Y.extend(ys[:1])
    real_Y.extend(fit_Y[1:])

    real_Y = np.array(real_Y)/np.sum(real_Y)

    ### 保存x和y

    age_dis = {}

    age_dis['x'] = fit_X
    age_dis['y'] = list(real_Y)

    open('rs_dis.json','w').write(json.dumps(age_dis))

    print 'powlaw saved to rs_dis.json'

    acc_Y = np.array([np.sum(real_Y[:i+1]) for i in range(len(real_Y))])

    ccdf_Y = 1- acc_Y

    pcts = []
    lpct = 1
    for i,cy in enumerate(ccdf_Y):
        # print cy,lpct
        pct = cy/lpct
        pcts.append(pct)
        lpct = cy
    print pcts

    l2=ax1.plot(fit_X[:-1],ccdf_Y[:-1],label=u'互补累积分布 $P_c(t)$',c='r')
    l1 = ax1.plot(fit_X[:-1],pcts[:-1],'--',label='$p_c(s_i|t)$')

    ax1.set_ylabel(u'概率', fontproperties='SimHei')
    # ax1.set_yscale('log')
    ax1.set_xlabel(u'年份($t$)',fontproperties='SimHei')

    print popt
    lns = l1+l2
    labels = [l.get_label() for l in lns]
    plt.legend(lns,labels,prop={'family':'SimHei','size':8})
    plt.tight_layout()
    plt.savefig('fig/continue_fit.png',dpi=400)

    print 'left prop saved to fig/continue_fit.png.'


    plt.figure(figsize=(5,4))
    for left in sorted(left_percent.keys()):
        percents = left_percent[left]

        ## moving average
        percents = np.convolve(percents, np.ones((10,))/10, mode='valid')

        plt.plot(years[9:],percents,label='t={:}'.format(left))

    plt.plot([1980]*10,np.linspace(0.001,1,10),'--',linewidth=0.5,c='r')

    plt.xlabel(u'年份(y)',fontproperties='SimHei')
    plt.ylabel(u'p_{rs}(t|y)')
    plt.yscale('log')

    art = plt.legend(prop={'family':'SimHei','size':8},loc=9,bbox_to_anchor=(0.5, -0.15), ncol=5)

    plt.tight_layout()

    plt.savefig('fig/left_percentage.png',dpi=400,additional_artists=[art],bbox_inches="tight")

    print 'author left probability saved to left_percentage.png '


    print 'author used %d, out interval %d' % (used_au_num,out_interval)

    plt.figure(figsize=(5,4))

    xs =[]
    ys =[]
    t=0
    ic = Counter(intervals)
    for n in sorted(ic.keys()):
        xs.append(n)
        if n==10:
            t1=t
        t+=ic.get(n)
        ys.append(t)


    print t
    ys = np.array(ys)/float(t)

    plt.plot(xs,ys)
    plt.xlabel(u'论文发表间隔(年)',fontproperties='SimHei')
    plt.ylabel(u'累积概率',fontproperties='SimHei')
    plt.plot([10]*10,np.linspace(0.5,1,10),'--',c='r')
    # plt.text(10,0.9,'97%')
    plt.text(11, 0.95, str('(10, {:.1%})'.format(t1/float(t))))
    # plt.arrow(12,0.9,-2,t1/float(t)-0.9, head_width=0.05, head_length=0.2, fc='k', ec='k')
    # plt.yscale('log')

    params = {'legend.fontsize': 5,
        'axes.labelsize': 5,
        'axes.titlesize':5,
        'xtick.labelsize':5,
        'ytick.labelsize':5}

    pylab.rcParams.update(params)
    a = plt.axes([.6, .2, .35, .35])

    plt.hist(intervals,33,rwidth=0.5,density=True)
    mean = np.mean(intervals)
    median = np.median(intervals)

    # print matplotlib.matplotlib_fname()
    print mean,median
    plt.title(u'密度曲线', fontproperties='SimHei')

    # plt.xlabel(u'发表间隔(年)', fontproperties='SimHei')
    # plt.ylabel(u'比例', fontproperties='SimHei')
    plt.yscale('log')

    plt.tight_layout()
    plt.savefig('fig/interval.png',dpi=400)
    print 'interval distribution saved to fig/interval.png'
    # plt.legend(prop={'family':'SimHei','size':15})
    # return

    params = {'legend.fontsize': 10,
        'axes.labelsize': 10,
        'axes.titlesize':15,
        'xtick.labelsize':10,
        'ytick.labelsize':10}

    pylab.rcParams.update(params)

    ### 根据这个时间画图
    print 'plot year related num figure ...'
    years = []
    news = []
    ends = []
    total = 0
    tlist = []
    for year in sorted(year_new_num.keys()):

        if year>2006:
            continue

        years.append(year)

        new_num = year_new_num[year]
        end_num = year_end_num.get(year,0)

        # print year,total,new_num,end_num

        total+=new_num-end_num

        tlist.append(total)
        news.append(new_num)
        ends.append(end_num)

    plt.figure(figsize=(5,4))
    plt.plot(years,news,label=u'新作者数')
    plt.plot(years,ends,'--',label=u'离开作者数')
    plt.plot(years,tlist,label=u'剩余作者总数')
    plt.legend(prop={'family':'SimHei','size':10})

    plt.xlabel('年份',fontproperties='SimHei')
    plt.ylabel('作者数量',fontproperties='SimHei')
    plt.yscale('log')


    # plt.legend()
    plt.tight_layout()
    plt.savefig('fig/author_num.png',dpi=400)

    ### 作者数量拟合 news
    plt.figure(figsize=(5,4))

    # ts = np.array(years)-years[0]+1
    ts = np.array(years)-years[0]+1
    print news[0]

    plt.plot(ts,news,'o',mec='#D2D2D2',mew=0.5,label=u'新作者数')
    # print
    ## 使用指数函数进行拟合
    expfunc = lambda t,a,b:news[0]*a*np.exp(b*t)
    popt, pcov = scipy.optimize.curve_fit(expfunc,ts,news,p0=(2,0.01))

    fit_Y = [expfunc(t,*popt) for t in ts ]

    print ts[:10]
    print fit_Y[:10]
    print 'exponential:',popt

    plt.plot(ts,fit_Y,'--',label=u'拟合曲线 $s_n(t) = s_0*%.2f*e^{%.2ft}$'%(popt[0],popt[1]),c='r')

    plt.xlabel(u'年份$t$',fontproperties='SimHei')
    plt.ylabel(u'新作者数量$s_n(t)$',fontproperties='SimHei')
    plt.legend(prop={'family':'SimHei','size':8})
    plt.yscale('log')
    plt.tight_layout()

    plt.savefig('fig/author_num_fit.png',dpi=400)

    print 'fiting author num figure saved to fig/author_num_fit.png'



    return
    ### lifespan的图
    print 'plot lifespan related figure ...'
    xs = []
    ys = []
    nums = []
    for lifespan in sorted(lifespan_dis.keys()):
        xs.append(lifespan)
        ys.append(lifespan_dis[lifespan])
        nums.append(lifespan_nums[lifespan])

    ages = []
    anums = []

    for age in sorted(age_num.keys()):
        ages.append(age)
        anums.append(age_num[age])


    fig,axes = plt.subplots(6,1,figsize=(6,30))

    ## 研究寿命分布的关系
    ax0 = axes[0]

    ax0.plot(xs,ys)
    ax0.set_xlabel('lifespan')
    ax0.set_ylabel('number of authors')
    ax0.set_yscale('log')
    ax0.set_xscale('log')


    ax1 = axes[1]
    ys = []
    for i,num in enumerate(nums):

        ys.append(np.mean(num))

        ax1.scatter([xs[i]]*len(num),num)

    ax1.plot(xs,ys)

    ax1.set_xlabel('lifespan')
    ax1.set_ylabel('number of papers')

    ax2 = axes[2]
    ax5 = axes[4]
    ax6 = axes[5]
    ys = []
    for i,num in enumerate(anums):
        ax2.scatter([ages[i]]*len(num),num)
        ys.append(np.mean(num))

        if i==0:
            ax5.hist(num,10)
        elif i==5:
            ax6.hist(num,10)


    ax2.set_xlabel('age')
    ax2.plot(ages,ys)
    ax2.set_ylabel('number of papers')

    ax3 = axes[3]

    xs = []
    ys = []

    for an in sorted(article_num_dis.keys()):
        xs.append(an)
        ys.append(article_num_dis[an])

    ax3.plot(xs,ys)
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_xlabel('number of article')
    ax3.set_ylabel('number of authors')


    plt.tight_layout()
    plt.savefig('fig/lifespan_num.png',dpi=400)

    print 'done'


### 价值增益分布函数的图
def ID_plot():
    t = np.arange(0, 10, 0.1)
    d = np.exp(-1)*np.power(1, t)/factorial(t)

    plt.figure(figsize=(4.2,4))
    plt.plot(t,d,c='b',linewidth=1)

    plt.xlabel('价值增益系数$I(D)$')
    plt.ylabel('概率$P$')

    plt.tight_layout()

    plt.savefig('fig/ID_dis.png',dpi=400)



def compare_plots():
    author_year_articles = json.loads(open(AUTHOR_JSON_PATH).read())

    ## 所有作者的文章总数量
    tnas = []

    ## 领域内作者总数量
    year_an = defaultdict(int)

    ## 领域内文章总数量
    year_pn = defaultdict(list)

    ## 对于每一位作者来讲
    for author in author_year_articles.keys():

        total_num_of_articles = 0
        ## 每一年
        for i,year in enumerate(sorted(author_year_articles[author].keys(),key=lambda x:int(x))):

            ##第一年是作者进入的年
            if i==0:
                year_an[int(year)]+=1

            ## 文章数量
            num_of_articles = len(author_year_articles[author][year])

            total_num_of_articles+=num_of_articles

            year_pn[int(year)].append(num_of_articles)

        tnas.append(total_num_of_articles)

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
    plt.title('APS')
    # plt.xscale('log')
    plt.yscale('log')
    plt.legend(prop={'family':'SimHei','size':8})

    plt.tight_layout()
    plt.savefig('fig/compare_pn.png',dpi=400)
    print 'simulation of total papers saved to fig/compare_pn.png'

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
    plt.title('APS')
    plt.tight_layout()
    plt.savefig('fig/compare_tn.png',dpi=400)
    print 'data saved to fig/compare_tn.png'



if __name__ == '__main__':

    # ## 生成数据
    # extract_from_metadata()
    # ## 画图
    # delta_num()

    # author_productivity()
    compare_plots()

    # ID_plot()


































