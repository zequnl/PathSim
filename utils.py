import numpy as np
def load_data():
    '''
    数据处理
    :return: author_info：字典，键为作者在原数据集编号，值为{"num":从0开始的新编号, "name"：作者姓名}的字典
              paper_info，venue_info同上
              num_author_map：字典，键为作者新编号，值为作者原编号，为了方便通过算法中使用的新编号查找到作者具体信息
              num_paper_map，num_venue_map同上
              paper_author_adj：numpy数组，paper和author的邻接矩阵
              paper_venue_adj：同上
    '''
    print("loading data...")
    f1 = open("data/author.txt")
    f2 = open("data/paper.txt")
    f3 = open("data/relation.txt")
    f4 = open("data/venue.txt")
    author_info = {}
    paper_info = {}
    venue_info = {}
    num_author_map = {}
    num_paper_map = {}
    num_venue_map = {}
    author_num = 0
    paper_num = 0
    venue_num = 0
    for line in f1:
        strs = line.split("\t")
        info = {}
        info["num"] = author_num
        info["name"] = strs[1]
        author_info[strs[0]] = info
        num_author_map[str(author_num)] = strs[0]
        author_num += 1
    for line in f2:
        strs = line.split("\t")
        info = {}
        info["num"] = paper_num
        info["name"] = strs[1]
        paper_info[strs[0]] = info
        num_paper_map[str(paper_num)] = strs[0]
        paper_num += 1
    for line in f4:
        strs = line.split("\t")
        info = {}
        info["num"] = venue_num
        info["name"] = strs[1]
        venue_info[strs[0]] = info
        num_venue_map[str(venue_num)] = strs[0]
        venue_num += 1
    paper_author_adj = np.zeros((paper_num, author_num))
    paper_venue_adj = np.zeros((paper_num, venue_num))
    for line in f3:
        strs = line.split("\t")
        if strs[0] in paper_info:
            idx = paper_info[strs[0]]["num"]
            if strs[1] in author_info:
                num = author_info[strs[1]]["num"]
                paper_author_adj[idx][num] += 1
            if strs[1] in venue_info:
                num = venue_info[strs[1]]["num"]
                paper_venue_adj[idx][num] += 1
    print("Dataset information:")
    print("authors:" + str(author_num))
    print("papers:" + str(paper_num))
    print("venues:" + str(venue_num))
    return author_info, paper_info, venue_info, num_author_map, num_paper_map, num_venue_map, paper_author_adj, \
           paper_venue_adj
