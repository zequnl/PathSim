import numpy as np
import math

class Cluster(object):
    '''
    重聚类的实现，row cluster_j的特征向量第i维是p(column cluster i | row cluster j)
    row_j的特征向量第i维是p(column cluster i | row_j)，每个row被分配到与其特征向量距离
    最近的row cluster， column cluster同理，反复迭代
    '''
    def __init__(self, m, n_cluster):
        self.m = m
        self.n_cluster = n_cluster
        r = []
        c = []
        row_num = len(m)
        col_num = len(m[0])
        row_batch = row_num // n_cluster + 1
        col_batch = col_num // n_cluster + 1
        '''
        对行和列的cluster初始化，先均匀分为n_cluster个类
        '''
        for i in range(n_cluster):
            row_end = (i + 1) * row_batch
            if row_end > row_num:
                row_end = row_num
            col_end = (i + 1) * col_batch
            if col_end > col_num:
                col_end = col_num
            r.append(list(range(i * row_batch, row_end)))
            c.append(list(range(i * col_batch, col_end)))
        iters = 3
        for t in range(iters):
            new_r = []
            new_c = []
            for i in range(n_cluster):
                new_r.append([])
                new_c.append([])
            '''
            列的重新分配
            '''
            rows_p = []
            for i in range(n_cluster):
                row_p = []
                sum = np.sum(m[r[i]])
                rows = m[r[i]]
                for j in range(n_cluster):
                    sub = rows[:, c[j]]
                    p = np.sum(sub)
                    if sum == 0:
                        row_p.append(0)
                        continue
                    row_p.append(p / sum)
                rows_p.append(row_p)
            for i in range(row_num):
                row = m[i]
                sum = np.sum(row)
                p = []
                for j in range(n_cluster):
                    sub = row[c[j]]
                    pt = np.sum(sub)
                    if sum == 0:
                        p.append(0)
                        continue
                    p.append(pt / sum)
                min1 = 10
                cluster = 0
                for j in range(n_cluster):
                    p_cluster = rows_p[j]
                    dist = np.linalg.norm(np.array(p) - np.array(p_cluster))
                    if dist < min1:
                        min1 = dist
                        cluster = j
                new_r[cluster].append(i)
            '''
            行的重新分配
            '''
            cols_p = []
            n = np.array(m.T)
            for i in range(n_cluster):
                col_p = []
                sum = np.sum(n[c[i]])
                cols = n[c[i]]
                for j in range(n_cluster):
                    sub = cols[:, r[j]]
                    p = np.sum(sub)
                    if sum == 0:
                        col_p.append(0)
                        continue
                    col_p.append(p / sum)
                cols_p.append(col_p)
            for i in range(col_num):
                colu = n[i]
                sum = np.sum(colu)
                p = []
                for j in range(n_cluster):
                    sub = colu[r[j]]
                    pt = np.sum(sub)
                    if sum == 0:
                        p.append(0)
                        continue
                    p.append(pt / sum)
                min1 = 10
                cluster = 0
                for j in range(n_cluster):
                    p_cluster = cols_p[j]
                    dist = np.linalg.norm(np.array(p) - np.array(p_cluster))
                    if dist < min1:
                        min1 = dist
                        cluster = j
                new_c[cluster].append(i)
            r = new_r
            c = new_c
        self.rows = r
        self.cols = c
        self.row_labels_ = np.zeros(row_num)
        self.column_labels_ = np.zeros(col_num)
        for i in range(len(self.rows)):
            for j in range(len(self.rows[i])):
                row = self.rows[i][j]
                self.row_labels_[row] = i
        for i in range(len(self.cols)):
            for j in range(len(self.cols[i])):
                col = self.cols[i][j]
                self.column_labels_[col] = i

    def get_submatrix(self, i, data):
        '''
        找到第i个块对应的子矩阵
        :param i: 块的编号
        :param data: 原矩阵
        :return: 子矩阵
        '''
        row = i // self.n_cluster
        col = i % self.n_cluster
        b = data[self.rows[row]]
        sub = b[:, self.cols[col]]
        return sub

    def get_indices(self, i):
        '''
        找到第i个块对应的行和列编号
        :param i: 块的编号
        :return: 一个数组，第一维是行编号的列表，第二位是列编号的列表
        '''
        row = i // self.n_cluster
        col = i % self.n_cluster
        r = []
        r.append(self.rows[row])
        r.append(self.cols[col])
        return r



