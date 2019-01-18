#coding:utf-8
'''
领域内作者数量，文章数量，引文网络记录

作者：作者ID，作者论文
文章：文章ID，文章价值增益
引文关系：引文关系增益

'''
import sys
sys.path.extend(['.','..'])
from tools.basic_config import *
import commons
from scholar import Scholar
import time
import copy


class Domain:

    def __init__(self):

        ## 领域内作者
        self._authors = []
        self._papers = []
        self._paper_ids = []
        self._paper_years = []
        self._paper_kgs = []
        ## 领域内引用关系,结构（citing_pid,cited_pid）
        self._crs = []
        ## 作者文章关系,初始化也是空的
        self._author_papers = defaultdict(list)

        ## 当前年份，最开始设置为1年，也就是该作者该没有开始写论文
        self._year = 1

        #初始化只有一个author,研究年龄为1,进入年份为1
        first_author = ('S_'+commons.gen_id(),1,1)
        # 内部结构（paperID,year, knowledge_gain],从无到有的增益为100，以后的论文以此为基础进行
        first_paper = ['P_'+commons.gen_id(),1,100]

        self._paper_ids.append(first_author[0])
        self._paper_years.append(1)
        self._paper_kgs.append(100)

        ## 初始化一篇文章,
        self._papers.append(first_paper)
        self._authors.append(first_author)
        self._author_papers[first_author[0]].append(first_paper[0])

        logging.info('initialize Domain parameters ...')


    ## 首先假设所有论文都可以被看到
    def view_papers(self):
        ## 根据领域内的kg算出领域内的kg_prob
        kg_probs = commons.kg_prob(self._paper_kgs)
        ## scale 到0-1之后，然后再进行normlize
        self._kg_probs = list(np.array(kg_probs)/np.sum(kg_probs))
        return copy.copy(self._paper_ids),copy.copy(self._paper_years),copy.copy(self._paper_kgs),self._kg_probs

    ## 将数据进行保存
    def save_data(self,data_path):
        data = {}
        data['authors'] = self._authors
        # data['crs'] = self._crs
        data['papers'] = self._papers
        data['author_papers'] = self._author_papers
        data['year'] = self._year
        open(data_path,'w').write(json.dumps(data))
        logging.info('data saved to {:}'.format(data_path))

    def save_crs(self,crs_path):
        f = open(crs_path,'w+')
        lines = []
        for cr in self.crs:
            line = ','.join(cr)
            lines.append(line)

            if len(lines)==10000:
                f.write('\n'.join(lines)+'\n')
                lines = []
        if len(lines)!=0:
            f.write('\n'.join(lines)+'\n')
        f.close()


    def load_data(self,data_path):
        data = json.loads(open(data_path).read())
        self._authors = data['authors']
        # self._crs = data['crs']
        self._papers = data['papers']
        self._author_papers = data['author_papers']
        self._year = data['year']

    ## 经过一年
    def one_year_later(self):
        ## 年份增加一年
        self._year+=1
        ## 领域内作者人数增加
        self._authors.extend([(a,1,self._year) for a in commons.add_authors(len(self._authors))])
        ## 这里面假设每个人看到的paper相同
        viewed_papers = self.view_papers()
        ## 存在的作者写论文
        for author in self._authors:
            aid = author[0]
            age = author[1]
            scholar = Scholar(age,viewed_papers)
            ## scholar 写论文
            for pid,ref_list,kg in scholar.write_papers():

                # self._papers.append([pid,self._year,kg])
                self._paper_ids.append(pid)
                self._paper_years.append(self._year)
                self._paper_kgs.append(kg)

                self._author_papers[aid].append(pid)
                self._crs.extend([[pid,ref_id] for ref_id in ref_list])

    def go_to_year(self,N):
        logging.info('let us go to {:} years later...'.format(N) )

        ## 文章数量的数组
        num_of_papers = [len(self._papers)]
        ## 作者数量
        num_of_authors = [len(self._authors)]
        ## citation rations
        num_of_crs = [len(self._crs)]
        ## citation distribution

        ## year
        years = [self._year]


        for _ in range(N):

            logging.info('year {:} has {:} papers and {:} authors.'.format(self._year,len(self._paper_ids),len(self._authors)))
            self.one_year_later()
            ## 画这一年的数据

            ## 文章数量
            num_of_papers.append(len(self._papers))

            num_of_authors.append(len(self._authors))

            num_of_crs.append(len(self._crs))

            years.append(self._year)

        # plt.show()

        fig,axes = plt.subplots(2,2,figsize=(10,10))

        ax0 = axes[0,0]
        ax0.plot(years,num_of_papers)
        ax0.set_xlabel('year')
        ax0.set_ylabel('number of papers')


        ax1 = axes[0,1]
        ax1.plot(years,num_of_authors)
        ax1.set_xlabel('year')
        ax1.set_ylabel('number of authors')

        ax2 = axes[1,0]
        ax2.plot(years,num_of_crs)
        ax2.set_xlabel('year')
        ax2.set_ylabel('number of crs')

        ax3 = axes[1,1]

        pid_count = Counter(zip(*self._crs)[1])

        cnum_dict  = defaultdict(int)
        for pid in pid_count.keys():
            cnum_dict[pid_count[pid]] +=1

        xs = []
        ys = []
        for cnum in sorted(cnum_dict.keys()):
            xs.append(cnum)
            ys.append(cnum_dict[cnum])


        ax3.plot(xs,ys)
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.set_xlabel('citation count')
        ax3.set_ylabel('number of papers')

        plt.tight_layout()

        plt.savefig('test.png',dpi=300)

















