#coding:utf-8
'''
价值函数生成
1. 参考价值与引用次数相关性分析
2. 相同参考价值下的论文价值分布
3. 价值系数拟合以及仿真

'''
import sys
sys.path.extend(['.','..'])
from tools.basic_config import *

from scipy.stats import pearsonr

import statsmodels.api as sm
import powerlaw

### 首先生成APS的引用数据
CITATION_DATA_PATH = '/Users/huangyong/Workspaces/Study/datasets/APS/aps-dataset-citations-2016/aps-dataset-citations-2016.csv'
DATA_PATH = '/Users/huangyong/Workspaces/Study/datasets/APS/aps-dataset-metadata-2016'


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

## 记录每篇论文的年份
def record_paper_year():
    pid_year = {}

    progress = 0
    for article_path in list_metadata():

        progress+=1

        if progress%10000==1:
            print 'progress,',progress

        article_json = json.loads(open(article_path).read())
        pid,authors,date,atype,affiliations = extract_author_info(article_json)

        if authors=='-1' or len(authors)==0 or date =='-1' or atype!='article' or len(authors)>10:
            continue

        year = int(date.split('-')[0])

        pid_year[pid] = year

    print '%d papers has year attr.' % len(pid_year)

    open('data/paper_year.json','w').write(json.dumps(pid_year))

    print 'data saved to data/paper_year.jon'




def gen_dataset():

    pid_year = json.loads(open('data/paper_year.json').read())

    pid_cn = defaultdict(int)

    pid_refs = defaultdict(list)

    pid_all_cn = defaultdict(int)

    for line in open(CITATION_DATA_PATH):
        line = line.strip()

        if line=='citing_doi,cited_doi':
            continue


        citing_pid,cited_pid = line.split(',')

        pid_all_cn[cited_pid]+=1


        citing_year = int(pid_year.get(citing_pid,-1))
        cited_year = int(pid_year.get(cited_pid,-1))

        if citing_year==-1 or cited_year==-1:
            continue

        if cited_year >2006:
            continue

        if citing_year-cited_year<=10:
            # print citing_year,cited_year
            pid_cn[cited_pid]+=1

        pid_refs[citing_pid].append(cited_pid)

    open('data/pid_all_cn.json','w').write(json.dumps(pid_all_cn))
    print 'data saved to data/pid_all_cn.json'


    saved_pid_refs= {}
    for pid in pid_refs.keys():
        ref_num = len(pid_refs[pid])

        if ref_num<20:
            continue

        has_0 = False
        for ref in pid_refs[pid]:

            if pid_cn.get(ref,0)==0:
                has_0=True

                break

        if has_0:
            continue

        saved_pid_refs[pid] = pid_refs[pid]


    open('data/pid_cn.json','w').write(json.dumps(pid_cn))
    print '%d papers reserved, and saved to data/pid_cn.json' % len(pid_cn.keys())

    open('data/pid_refs.json','w').write(json.dumps(saved_pid_refs))
    print '%d papers reserved, and saved to data/pid_refs.json' % len(saved_pid_refs.keys())


def ref_cit_relations():

    pid_refs = json.loads(open('data/pid_refs.json').read())
    pid_cn = json.loads(open('data/pid_cn.json').read())

    refvs = []
    c10l = []

    ref_c10s = defaultdict(list)

    v_coef_list = []

    for pid in pid_refs.keys():

        refs = pid_refs[pid]

        c10 = int(pid_cn.get(pid,0))

        if c10==0:
            continue


        ref_v = np.mean([int(pid_cn.get(ref,0)) for ref in refs])

        # if ref_v<10:
        v_coef_list.append(float('{:.3f}'.format(c10/float(ref_v))))

        ref_c10s[int(ref_v)].append(c10)


        c10l.append(c10)
        refvs.append(ref_v)


    ## 对价值系数进行估计

    vc_dict = Counter(v_coef_list)
    xs = []
    ys = []
    for vc in sorted(vc_dict.keys()):
        xs.append(vc)
        ys.append(vc_dict[vc])


    fit=powerlaw.Fit(np.array(v_coef_list),discrete=True,xmin=1)

    print 'xmin:',fit.power_law.xmin
    print 'alpha:',fit.power_law.alpha
    print 'sigma:',fit.power_law.sigma

    print 'compare:',fit.distribution_compare('power_law', 'exponential')
    print 'compare:',fit.distribution_compare('truncated_power_law', 'exponential')
    print 'compare:',fit.distribution_compare('lognormal', 'exponential')


    ## curve_fit
    expfunc = lambda t,a,b:b*np.exp(a*t)

    ys = np.array(ys)/float(np.sum(ys))
    popt,pcov = scipy.optimize.curve_fit(expfunc,xs,ys,p0=(0.1,-1))

    print popt

    print scipy.stats.lognorm._fitstart(v_coef_list)

    print scipy.stats.lognorm.fit(v_coef_list,loc=0)

    param=scipy.stats.lognorm.fit(v_coef_list,floc=0)

    print param
    # print np.exp(param[2])



    pdf_fitted = scipy.stats.lognorm.pdf(xs, param[0], loc=param[1], scale=param[2])

    pdf_fitted = np.array(pdf_fitted)/np.sum(pdf_fitted)


    lambda_dis = {}

    lambda_dis['x'] = xs
    lambda_dis['y'] = list(pdf_fitted)

    open('data/lambda_dis.json','w').write(json.dumps(lambda_dis))

    print 'data saved to data/lambda_dis.json'


    plt.figure(figsize=(5,4))
    plt.plot(xs,ys,'o')
    # plt.hist(v_coef_list,bins=100,density=True,histtype='step')
    fit_x = np.linspace(0.1,6,100)

    plt.plot(xs,pdf_fitted,'--',linewidth=3,label=u'拟合曲线$loc=0,scale=%.2f,\sigma=%.2f$' %(param[2],param[0]))

    x0=xs[list(pdf_fitted).index(np.max(pdf_fitted))]

    print 'x0:',x0

    plt.plot([x0]*10,np.linspace(0.000001,0.005,10),'-.',label='$\lambda_{0}=%.3f$'%x0,c='r')

    plt.yscale('log')
    plt.xscale('log')

    plt.xlabel('$\lambda$')
    plt.ylabel('$P(\lambda)$')
    plt.legend(prop={'family':'SimHei','size':8})

    plt.tight_layout()

    plt.savefig('fig/lambda_fit.png',dpi=400)

    print 'lambda fit figure saved to lambda_fit.png'

    ## 相同的参考价值之上的真实价值分布
    plt.figure(figsize=(5,4))
    # styles = ['-']
    for i,refv in enumerate([10,20,40,50,60,80]):

        c10_list = ref_c10s[refv]

        c10_dis = Counter(c10_list)

        xs = []
        ys = []

        for c10 in sorted(c10_dis.keys()):
            xs.append(c10)
            ys.append(c10_dis[c10])

        ys = np.array(ys)/float(np.sum(ys))

        # ax = axes[i/3,i%3]
        # xs = range(1,len(c10_list)+1)
        plt.plot(xs,ys,label=u'$\widehat{\langle v(refs) \\rangle}$=%d'%refv )

    plt.yscale('log')

    plt.xlabel(u'$c_{10}$',fontproperties='SimHei')
    plt.ylabel(u'比例',fontproperties='SimHei')
    plt.xscale('log')

    # plt.set_xlim(-1,7)
    # plt.xticks(range(len(xs)))
    # plt.set_xticklabels([int(x) for x in xs])
    # plt.spines['top'].set_visible(False)
    # plt.spines['right'].set_visible(False)
    # ax.set_xlim(-1,19)

    # ax.legend(loc=2)
    plt.legend(prop={'family':'SimHei','size':8})


    plt.tight_layout()

    plt.savefig('fig/refv_c10_dis.png',dpi=400)

    print 'fig saved to fig/refv_c10_dis.png'

    ##在同样基础之上的分布
    plt.figure(figsize=(5,4))
    # styles = ['-']
    for i,refv in enumerate([10,20,40,50,60,80]):

        c10_list = ref_c10s[refv]

        c10_dis = Counter(c10_list)

        xs = []
        ys = []

        for c10 in sorted(c10_dis.keys()):
            xs.append(c10/float(refv))
            ys.append(c10_dis[c10])

        ys = np.array(ys)/float(np.sum(ys))

        expfunc = lambda t,a,b:b*np.exp(a*t)

        popt,pcov = scipy.optimize.curve_fit(expfunc,xs,ys,p0=(0.1,-1))

        print popt
        # ax = axes[i/3,i%3]
        # xs = range(1,len(c10_list)+1)
        plt.plot(xs,ys,label=u'$\widehat{\langle v(refs) \\rangle}$=%d'%refv )

    plt.yscale('log')

    plt.xlabel(u'价值系数$\lambda$',fontproperties='SimHei')
    plt.ylabel(u'比例',fontproperties='SimHei')
    plt.xscale('log')

    plt.xlim(-1,7)
    # plt.xticks(range(len(xs)))
    # plt.set_xticklabels([int(x) for x in xs])
    # plt.spines['top'].set_visible(False)
    # plt.spines['right'].set_visible(False)
    # ax.set_xlim(-1,19)

    # ax.legend(loc=2)
    plt.legend(prop={'family':'SimHei','size':8})


    plt.tight_layout()

    plt.savefig('fig/lambda_dis.png',dpi=400)

    print 'fig saved to fig/lambda_dis.png'


    ps =  pearsonr(c10l,refvs)[0]

    print ps

    X = sm.add_constant(c10l)
    model = sm.OLS(refvs,X)
    res = model.fit()
    print(res.summary())

    linear_func = lambda x,a,b:a*x+b
    popt,pcov = scipy.optimize.curve_fit(linear_func,c10l,refvs)
    print popt


    plt.figure(figsize=(5,4))
    plt.plot(c10l,refvs,'o',alpha=0.7,label='皮尔逊相关系数:%.4f' % float(ps))
    xs = range(1,2000)
    ys = [linear_func(x,*popt) for x in xs ]
    plt.plot(xs,ys,'--',linewidth=3,label=u'y=%.3fx+%.3f,$R^2=$%.3f'%(popt[0],popt[1],0.031))
    plt.xlabel('$c_{10}$')
    plt.ylabel('$\langle v(refs) \\rangle$')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(prop={'family':'SimHei','size':8})
    plt.tight_layout()

    plt.savefig('fig/c10_ref_relations.png',dpi=400)

    print 'relation fig saved to fig/c10_ref_relations.png'


## 51万
def compare_citation_dis():

    pid_cn = json.loads(open('data/pid_all_cn.json').read())

    print len(pid_cn.keys())

    values = pid_cn.values()

    fit = powerlaw.Fit(values,xmin=1)

    print fit.power_law.xmin

    print 'compare:',fit.distribution_compare('power_law', 'exponential')

    cn_counter = Counter(values)

    xs = []
    ys = []
    for cn in sorted(cn_counter.keys()):
        xs.append(cn)
        ys.append(cn_counter[cn])

    plt.figure(figsize=(5,4))

    ys = np.array(ys)/float(np.sum(ys))

    plt.plot(xs,ys)

    plt.xlabel('$\#(c_i)$')
    plt.ylabel('$p(c_i)$')
    plt.title('APS')
    plt.xscale('log')

    plt.yscale('log')

    plt.tight_layout()

    plt.savefig('fig/compare_citation_dis.png',dpi=400)

    print 'fig saved to fig/compare_citation_dis.png.'

if __name__ == '__main__':
    # record_paper_year()

    # gen_dataset()

    # ref_cit_relations()

    compare_citation_dis()





