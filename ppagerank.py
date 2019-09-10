import numpy as np
import heapq

class PPageRank(object):
    '''
    personalized pagerank的实现
    '''
    def __init__(self, m, p):
        self.m = m  #邻接矩阵
        self.p = p
        self.d = np.zeros((len(m), len(m[0])))
        for i in range(len(m)):
            self.d[i][i] = 1 / np.sum(m[i])
        self.GD = np.mat(self.m) * np.mat(self.d)  #对邻接矩阵归一化，使其列和为1

    def find_topk(self, x, k):
        '''
        找到相似的前k个
        :param x: 查询对象编号
        :param k: 要找到的相似对象的数量
        :return: k个相似对象的list，按相似度降序排列
        '''
        t = np.zeros((len(self.m), len(self.m[0])))
        t[x][x] = 1  #将待查询的对象设置为personalized pagerank中感兴趣的主题
        v = []
        for i in range(len(self.m)):
            v.append(1 / len(self.m))
        iters = 5  #迭代
        for i in range(iters):
            A = self.p * self.GD + (1 - self.p) * np.mat(t)
            tmp = A * np.mat(v).T
            v = list(tmp.T)[0]
            #print(np.array(v)[0][222])
        #print(v)
        result = []
        result.append(x)
        topk = heapq.nlargest(k, range(len(np.array(v)[0])), np.array(v)[0].__getitem__)
        #print(topk)
        for i in topk:
            if i in result:
                continue
            if len(result) >= k:
                continue
            result.append(i)
        return result


