import numpy as np
import heapq
import sklearn
import bisect
import math
import co_cluster
from sklearn.cluster.bicluster import SpectralBiclustering

class Node(object):
    '''
    结点类，成员变量为每个待查询对象的编号和与目标对象的相似度
    '''

    def __init__(self, idx, val):
        self.idx = idx
        self.val = val

    def __lt__(self, other):
        return self.val < other.val


class PathSim(object):
    '''
    pathsim算法的实现类，元路径为P=(PlPl')，给定Pl的关系矩阵m和P的关系矩阵
    M对角线上的值，计算与查询对象相似的top-k对象
    '''

    def __init__(self, m, diag, dense=False):
        self.m = m  # Pl的关系矩阵m
        self.diag = diag  # P的关系矩阵M对角线上的值
        self.dense = dense
    def baseline(self, x, k):
        '''
        pathsim-baseline算法
        :param x: 查询对象编号
        :param k: 要找到的相似对象的数量
        :return: k个相似对象的list，按相似度降序排列
        '''
        candidate = []  # 找到候选对象
        if self.dense == False:
            for i in range(len(self.m[x])):
                if self.m[x][i] != 0:
                    for j in range(len(self.m)):
                        if self.m[j][i] != 0:
                            if j in candidate:
                                continue
                            candidate.append(j)
        else:
            for i in range(len(self.m)):
                candidate.append(i)  #稠密矩阵中，不需要通过邻居的邻居来寻找candidate
        mt = []  # 候选对象所在行向量组成矩阵
        for i in range(len(candidate)):
            mt.append(self.m[candidate[i]])
        mx1 = np.mat(self.m[x]) * np.mat(mt).T  # 矩阵乘法找到每个候选对象candidate[i]对应的M[x][i]
        mx = np.array(mx1)[0]
        # print(mx)
        sim = []
        for i in range(len(mx)):
            mxi = mx[i]
            mxx = self.diag[x]
            mii = self.diag[candidate[i]]
            s = 2 * mxi / (mxx + mii)
            sim.append(s)
        result = heapq.nlargest(k, range(len(sim)), sim.__getitem__)
        topk = []
        for i in range(len(result)):
            num = result[i]
            topk.append(candidate[num])
        return topk

    def pruning_init(self):
        '''
        剪枝算法前对矩阵进行分块和统计的预处理
        '''
        mt = np.array(np.mat(self.m).T)
        self.cluster_num = 3
        cluster_num = self.cluster_num
        self.model = co_cluster.Cluster(mt, self.cluster_num)  #重聚类
        print("co-clustering done")
        information = []
        for i in range(0, self.cluster_num * self.cluster_num):
            infor = {}
            sub_m = self.model.get_submatrix(i, mt)
            infor["t"] = np.sum(sub_m)
            infor["t1"] = np.sum(sub_m, axis=0)
            infor["tt1"] = np.sum(sub_m ** 2, axis=0)
            information.append(infor)
        self.T = np.zeros((cluster_num, cluster_num))  #每个块的元素和
        self.T1 = []  #每个块的列和
        self.TT1 = []  #每个块的列平方和
        for i in range(cluster_num):
            self.T1.append([])
            self.TT1.append([])
        for i in range(cluster_num * cluster_num):
            self.T[i // cluster_num][i % cluster_num] = information[i]["t"]
            self.T1[i // cluster_num] += list(information[i]["t1"])
            self.TT1[i // cluster_num] += list(information[i]["tt1"])
        self.new_T1 = np.array(self.T1)
        self.new_TT1 = np.array(self.TT1)

    def pruning(self, x, k):
        '''
        pathsim-pruning算法
        :param x: 查询对象编号
        :param k: 要找到的相似对象的数量
        :return: k个相似对象的list，按相似度降序排列
        '''
        candidate = []  # 找到候选对象
        if self.dense == False:
            for i in range(len(self.m[x])):
                if self.m[x][i] != 0:
                    for j in range(len(self.m)):
                        if self.m[j][i] != 0:
                            if j in candidate:
                                continue
                            candidate.append(j)
        else:
            for i in range(len(self.m)):
                candidate.append(i)  #稠密矩阵中，不需要通过邻居的邻居来寻找candidate
        cluster_num = self.cluster_num
        xt = self.m[x]
        row_label = self.model.row_labels_
        column_label = self.model.column_labels_
        x1 = []
        x2 = []
        for i in range(cluster_num):
            x1.append(0)
            x2.append(0)
        for i in range(len(xt)):
            cluster = int(row_label[i])
            if x1[cluster] < xt[i]:
                x1[cluster] = xt[i]
            x2[cluster] += xt[i] * xt[i]
        for i in range(cluster_num):
            x2[i] = math.sqrt(x2[i])
        upperbound = np.zeros(cluster_num)  #每个块的相似度上界
        for i in range(cluster_num):
            upper = 2 * np.mat(x1) * np.mat(self.T[:, i]).T / (self.diag[x] + 1)[0][0]
            upperbound[i] = upper
        #print(upperbound)
        cluster = np.argsort(-upperbound)
        s = []
        length = np.zeros(cluster_num)
        for i in cluster:
            column = self.model.get_indices(i)
            length[i] = len(column[1])
        bias = np.zeros(cluster_num)
        for i in range(1, cluster_num):
            bias[i] = bias[i - 1] + length[i - 1]
        for i in cluster:
            if len(s) >= k and upperbound[i] <= s[len(s) - k].val:
                break  #块剪枝
            column = self.model.get_indices(i)
            for id in range(0, len(column[1])):
                if column[1][id] in candidate:
                    j = id + bias[i]  #计算每个候选对象在重聚类后的矩阵中对应的列
                    upper = 2 * np.mat(x2) * np.mat(self.new_TT1[:, int(j)]).T / (self.diag[x] + self.diag[column[1][id]])[0][0]
                    if len(s) >= k:
                        if upper <= s[len(s) - k].val:
                            print("pruning:" + str(j))
                            continue  #计算相似度上界后进行剪枝
                    val = 2 * np.mat(xt) * np.mat(self.m[column[1][id]]).T / (self.diag[x] + self.diag[column[1][id]])[0][0]
                    bisect.insort(s, Node(column[1][id], val))
        topk = []
        if k > len(s):
            k = len(s)
        for i in range(k):
            topk.append(s[len(s) - 1 - i].idx)
        return topk