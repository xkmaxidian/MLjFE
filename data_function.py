
# coding: utf-8

# In[1]:


import networkx as nx
#import matplotlib.pyplot as plt
import pandas as pd
#import scipy.sparse as sp
import numpy as np
#import pickle

import pickle

# In[2]:

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve

def split_data(G_test, Adj_pred):
    edgelist_pos = [edges for edges in G_test.edges()]
    edges_del = []
    
    for i in range(len(edgelist_pos)):
        edges = edgelist_pos[i];
        if edges[0] not in Adj_pred.index  or edges[1] not in Adj_pred.index:
            #print edges
            edges_del.append(edges);
    for edges in edges_del:
        edgelist_pos.remove(edges)
    
    edgelist_neg = []
    i=0;
    #print G.number_of_edges()
    while i < 10*len(edgelist_pos):
        idx_i = np.random.randint(0, len(Adj_pred.index))
        idx_j = np.random.randint(0, len(Adj_pred.index))
        e = (Adj_pred.index[idx_i],Adj_pred.index[idx_j]);
        if idx_i==idx_j:
            continue;
        if e in edgelist_pos:
            continue;
        if e[0] not in G_test.nodes or e[1] not in G_test.nodes:
            continue;
        i = i+1;
        edgelist_neg.append(e);
    print "size of positive edges: ",len(edgelist_pos);
    print "size of negtive edges: ",len(edgelist_neg);
    return edgelist_pos,edgelist_neg;

def get_roc_score(edges_pos, edges_neg, score_matrix):
    # Store positive edge predictions, actual values
    preds_pos = []
    #pos = []
    for edge in edges_pos:
        preds_pos.append(score_matrix.loc[edge[0], edge[1]]) # predicted score
        #pos.append(adj_sparse[edge[0], edge[1]]) # actual value (1 for positive)
        
    # Store negative edge predictions, actual values
    preds_neg = []
    #neg = []
    for edge in edges_neg:
        preds_neg.append(score_matrix.loc[edge[0], edge[1]]) # predicted score
        #neg.append(adj_sparse[edge[0], edge[1]]) # actual value (0 for negative)
        
    # Calculate scores
    preds_all = np.hstack([preds_pos, preds_neg])
    labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    fpr, tpr, thresholds = roc_curve(labels_all, preds_all)
    return roc_score, ap_score, fpr, tpr, thresholds

def AUC_AP_Score(G_test, pred):
    Adj_test = nx.to_pandas_adjacency(G_test, dtype=int, multigraph_weight=sum)
    Adj_pred = pred
    edgelist_pos,edgelist_neg = split_data(G_test, Adj_pred);
    roc_score, ap_score, fpr, tpr, thresholds=get_roc_score(edgelist_pos, edgelist_neg, Adj_pred)
    return roc_score, ap_score, fpr, tpr, thresholds

## add noise to each Graph in Gs according to rate
def add_noise(Gs, rate):
    for i in range(len(Gs)):
        num_nodes = nx.number_of_nodes(Gs[i])
        num_edges = nx.number_of_edges(Gs[i])
        num_noises = int(rate*num_edges)
        nodes1 = np.random.randint(num_nodes, size=(num_noises,num_noises));
        nodes2 = np.random.randint(num_noises, size=(num_noises,num_noises));
        for n1 in nodes1:
            for n2 in nodes2:
                Gs[i].add_edge(n1,n2);
    return Gs;
def save_Gs2pickle(Gs, path):
    ## 保存运算结果
    mydb  = open(path, 'w')  
    pickle.dump(Gs, mydb)
    mydb.close()
def load_pickle2Gs(path):
    ## 读取运算结果
    mydb  = open(path, 'r')  
    Gs = pickle.load(mydb)
    mydb.close()
    return Gs
# In[3]:


# # split in positive_edgelist and negtive_edgelist
# # positive means the edge in adjacency matrix is 1
# # negtive means the edge in adjacency matrix is 0
# # return type,
# def split_pos_neg(G):
#     pos_edgelist=nx.edges(G);
#     # get negtive adjacency matrix which set the diagnol elements to 0
# #     adjacency_matrix=nx.to_numpy_matrix(G);#adjacency matrix in numpy.matrix
# #     adjacency_array=np.asarray(adjacency_matrix);#adjacency matrix in numpy array type
# #     neg_adj=np.abs(adjacency_array-1)
# #     neg_adj = neg_adj-np.diag(neg_adj.diagonal());
#     #neg_adj=np.matrix(neg_adj)
#     # convert from adjacency matrix to edgelist: 
#     ### way 1: convert to networkx graph type firstly, then convert to edgelist
#     #neg_G = nx.from_numpy_matrix(neg_adj);
#     #neg_edgelist = nx.edges(neg_G);
#     ### way 2: convert to edgelist directly
# #     neg_edgelist=[]
# #     neg_edge_tuple=()
# #     for i in range(0,np.shape(neg_adj)[0]):
# #         for j in range(i+1,np.shape(neg_adj)[0]):
# #             if neg_adj[i][j]==1:
# #                 neg_edge_tuple=(i,j)
# #                 neg_edgelist.append(neg_edge_tuple);
#     neg_edgelist=[]
#     max_nodeid = np.max(pos_edgelist);
#     # denote there is not node0 in the nodes table
#     for i in range(1,max_nodeid+1):
#         for j in range (i+1,max_nodeid+1):
#             if (i,j) not in pos_edgelist:
#                 neg_edgelist.append((i,j));
#     print "positive edges shape:",np.shape(pos_edgelist);
#     print "negtive edges shape:",np.shape(neg_edgelist);
#     return pos_edgelist,neg_edgelist;
# #print type(pos_edgelist),type(neg_edgelist)
# #print np.shape(neg_edgelist)



# convert from np.array adjacency matrix to edgelist
# just for undirected graph


# In[4]:


# read edgelist as graph type of networkx
import time;
import datetime;

filename="/home/tanshiyin/Documents/machineLearning/dynamicNetwork/CollegeMsg.txt";
def get_edgelist(filename):
    edgelist=np.loadtxt(filename);
    print "edgelist shape:",np.shape(edgelist);
    return edgelist;

# get graph from filename which is snap tmporal dataset
def get_graph_from_filename(filename):
    G = nx.read_edgelist(filename, nodetype=int, data=(('timestamp',int),))
    print nx.info(G)
    return G;
    # adjacency_matrix=nx.adj_matrix(G);#adjacency matrix in scipy.sparse.csr.csr_matrix
    # adjacency_array=adjacency_matrix.toarray();#adjacency matrix in numpy array type


# processing edgelist of filename
## processing unix timestamp to numpy.datetime64
#timestamp=edgelist[:,2];
# convert unix timestamp to numpy.datetime64
# time format is python time.strftime format
def convert_datetime(timestamp, time_format="%Y-%m-%d"):
    print "timestamps shape: ",(np.shape(timestamp))
    dates=[]
    # index of dates
    idx_dates=[]
    for i in range(0,np.shape(timestamp)[0]):
        # convers unix timestamp to numpy.datetime64
        #dates.append(np.datetime64(time.strftime(time_format,time.gmtime(timestamp[i]))));
        dates.append(time.strftime(time_format, time.gmtime(timestamp[i])))
        idx_dates.append(i)
    
    return idx_dates, dates;


#print datetime.datetime.fromtimestamp(timestamp[0])
# convert unix timestamp to datetime.datetime
#for i in range(0,np.shape(timestamp)[0]):
#    datetime.datetime.fromtimestamp(timestamp[i])

# save edgelist data to pandas.Series(edge,datetime.datetime);
# dates: which is a array of numpy.datetime64, see convert_datetime
# edgelist: see get_edgelist(filename)
def convert_Series(edgelist):
    # get dates
    timestamp=edgelist[:,2];
    dates=convert_datetime(timestamp);
    index=pd.DatetimeIndex(dates);
    edges=tuple(edgelist[:,0:2])
    datas_Series=pd.Series(edges,index=index);
    return datas_Series;
#print datas['2004-04'].shape
#print datas['2004-05'].shape
#print datas['2004-06'].shape
#print datas['2004-07'].shape
#print datas['2004-08'].shape
#print datas['2004-09'].shape
#print datas['2004-10'].shape
#print datas

# save edgelist as pandas.DataFrame()
# we could convert DataFrame to networkx graph directly
# by using nx.from_pandas_dataframe()
def convert_DataFrame(edgelist):
    # get dates
    timestamp=edgelist[:,2];
    dates=convert_datetime(timestamp);
    # set Series index
    index=pd.DatetimeIndex(dates);
    # get DataFrame
    SNode=pd.Series(edgelist[:,0],index=index);
    DNode=pd.Series(edgelist[:,1],index=index);
    # dict of Series
    d={'sourceNode':SNode,'destNode':DNode}
    datas_DF=pd.DataFrame(d)
    return datas_DF;

# convert DataFrame to networkx graph type
#G=nx.from_pandas_dataframe(datas_DF, 'sourceNode', 'destNode')
#print nx.info(G)


# In[5]:


# temporal edgelist(with timestamps, col0:source, col1:destination, col2:timestamp)
# convert timestamps in temporal edgelist to datetime64 in form %Y,%M,%D
def convert_TemporalGraph(edgelist):
    edgelist[:,2]=convert_datetime(edgelist[:,2])
    return edgelist;


# In[6]:

# convert adjacency matrix to edgelist
def Adj2Edgelist(adj_matrix):
    shape=np.shape(adj_matrix);
    L=[];
    for i in range(0, shape[0]):
        for j in range(i+1,shape[0]):
            if adj_matrix[i][j]==1:
                t=(i,j);
                L.append(t);
    return L;



# load several edgelists from files to one graph
def load_edgelists2Graph(filenames):
    for i,filename in enumerate(filenames):
        G = nx.read_edgelist(filename, nodetype=int)
        if i==0:
            edges=list(G.edges());
        else:
            edges[len(edges):len(edges)]=list(G.edges());
    Graph=nx.from_edgelist(edges)
    return Graph;

# count edges in many edgelists
def count_edges(filenames):
    count = 0;
    for i,filename in enumerate(filenames):
        f=open(filename,"r");
        for index, line in enumerate(f):
            count += 1
    print(count)
    return count;

# combininig several Graphs to one Graph
def combiningGs2G(Gs):
    overallG=nx.Graph();
    for i,G in enumerate(Gs):
        overallG.add_edges_from(G.edges);
    return overallG;


## save several graphs Gs to several file_#.txt
def Gs2txt(Gs, directory, files):
    for i,G in enumerate(Gs):
        np.savetxt(directory+files[i],np.array([e for e in G.edges]),fmt="%d");





