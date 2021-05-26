
# coding: utf-8

# In[5]:


import numpy as np;
import pandas as pd;
import networkx as nx;
from sklearn.decomposition import NMF
from numpy import linalg as LA
import math
import matplotlib.pyplot as plt
import time
import pickle


# In[6]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve

def split_data(G_test, Adj_pred):
    edgelist_pos = [edges for edges in G_test.edges()]
    edges_del = []
    
    for i in range(len(edgelist_pos)):
        edge = edgelist_pos[i];
        if edge[0] not in Adj_pred.index  or edge[1] not in Adj_pred.columns:
            #print edge
            edges_del.append(edge);
    for edge in edges_del:
        edgelist_pos.remove(edge)
    
    edgelist_neg = []
    i=0;
    #print G.number_of_edges()
    # static network test: i<len(edgelist_pos)
    # dynamic network test: i<10*len(edgelist_pos)
    while i < 4*len(edgelist_pos):
        idx_i = np.random.randint(0, len(Adj_pred.index))
        idx_j = np.random.randint(0, len(Adj_pred.columns))
        e = (Adj_pred.index[idx_i],Adj_pred.columns[idx_j]);
        if idx_i==idx_j:
            continue;
        if e in edgelist_pos:
            continue;
        if e[0] not in G_test.nodes or e[1] not in G_test.nodes:
            continue;
        i = i+1;
        edgelist_neg.append(e);
    print("size of positive edges: ",len(edgelist_pos));
    print("size of negtive edges: ",len(edgelist_neg));
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


# In[7]:


import nmf
from sklearn.decomposition import TruncatedSVD
## non-smoothed nmf
def test_nmf(Gs, dim_latent, max_time, iter_num):
    Ws=[]
    Hs=[]
    for i in range(len(Gs)):
        A = nx.to_pandas_adjacency(Gs[i])
        ## initialed by TSVD
        svd = TruncatedSVD(n_components=dim_latent, n_iter=50)
        U = svd.fit_transform(A.values)
        V = svd.components_
        W,H = nmf.nmf(A.values, U, V, 1e-4, max_time, iter_num)
        Ws.append(pd.DataFrame(W, index=A.index))
        Hs.append(pd.DataFrame(H, columns=A.columns))
    return Ws,Hs


# ## load data

# In[8]:


## add noise to each Graph in Gs according to rate
def add_noise(Gs, rate, p):
    for i in range(len(Gs)):
        num_nodes = nx.number_of_nodes(Gs[i])
        num_edges = nx.number_of_edges(Gs[i])
        num_noises = int(rate*num_edges)
        nodes1 = np.random.randint(num_nodes, num_noises);
        nodes2 = np.random.randint(num_nodes, num_noises);
        for n1 in nodes1:
            for n2 in nodes2:
                Gs[i].add_edge(n1,n2);


# In[9]:


# ## Cellphone data
# directory = "/home/tanshiyin/Documents/machineLearning/dynamicNetwork/temporalData/Cell/"
# filenames=["real.t01.edges","real.t02.edges","real.t03.edges","real.t04.edges","real.t05.edges", \
#           "real.t06.edges","real.t07.edges","real.t08.edges","real.t09.edges","real.t010.edges"]
# ### Cell data的换行符为'\r', 所以不能直接用nx.read_edgelist
# Gs=[]
# for i in range(0,len(filenames)):
#     a=pd.read_csv(directory+filenames[i],header=None,index_col=False,names=['source','target'], delimiter='\t')
#     G=nx.from_pandas_edgelist(a)
#     Gs.append(G)

    





# In[10]:


# smoothed NMF by alternative non-negative least squares using projected gradients
# Author: Chih-Jen Lin, National Taiwan University
# Python/numpy translation: Anthony Di Franco
## Projected Gradient Methods for Non-negative Matrix Factorization
from sklearn.decomposition import TruncatedSVD
from numpy import *
from numpy.linalg import norm
from time import time
from sys import stdout
from nmf import *
import gc;
from sklearn.decomposition import NMF




def sm_lnmf(Gs,dim_latent,lamb,tol=1e-3,timelimit=1000,maxiter=300,init="random"):
    gc.enable();
    num_graphs = len(Gs)
    Ws=[]
    Hs=[]
    for i in range(num_graphs):
        Adj = nx.to_pandas_adjacency(Gs[i]);
#         print "Graph %d "%i,"Shape:",Adj.shape
        ## find init that f(Winit,Hinit)<f(0,0)
        if init=="random":
            Winit = np.random.rand(len(Adj),dim_latent);
            Hinit = np.random.rand(dim_latent,len(Adj));
            #if LA.norm(Adj-Winit.dot(Hinit))>LA.norm(Adj):
                #print "init f is larger than f(0,0)!!";
                #model = NMF(n_components=dim_latent, init='random', random_state=0)
                #Winit = model.fit_transform(Adj.values)
                #Hinit = model.components_
                
            for idx in range(len(Adj)):
                Winit[idx,:] = Winit[idx,:]/LA.norm(Winit[idx,:])
                Hinit[:,idx] = Hinit[:,idx]/LA.norm(Hinit[:,idx])
        if init=="svd":
            ## initialed by TSVD
            svd = TruncatedSVD(n_components=dim_latent, n_iter=200)
            Winit = svd.fit_transform(Adj.values)
            Hinit = svd.components_
        if i==0:
            W,H,iter,projnorm = lnmf(Adj,Winit,Hinit,0,tol,timelimit,maxiter);
            Wpre = pd.DataFrame(W, index=Adj.index);
            Hpre = pd.DataFrame(H, columns=Adj.columns);
        else:
            drop = []
            ## V plus Wpre.dot(Hpre)
            for idx in Wpre.index:
                if idx not in Adj.index:
                    drop.append(idx);
            VpWH = Adj.add((lamb*(Wpre.dot(Hpre))), fill_value=0);
            VpWH = VpWH.drop(index=drop,columns=drop)
            if np.any(VpWH.isna()):
                print("None error");
            ## 由于要与上一时刻相加，所以Adj,Wpre,Hpre,使用pd.DataFrame结构
            W,H,iter,projnorm = lnmf(VpWH,Winit,Hinit,lamb,tol,timelimit,maxiter);
            Wpre = pd.DataFrame(W, index=VpWH.index);
            Hpre = pd.DataFrame(H, columns=VpWH.columns);
        print("Iteration: %d, Final proj-grad norm: %f"%(iter,projnorm));
#         del(W);del(H);
        Ws.append(Wpre)
        Hs.append(Hpre)
        gc.collect();
    return Ws,Hs;

def lnmf(V,Winit,Hinit,lamb,tol,timelimit,maxiter):
    """
    (W,H) = nmf(V,Winit,Hinit,tol,timelimit,maxiter)
    W,H: output solution
    Winit,Hinit: initial solution
    tol: tolerance for a relative stopping condition
    timelimit, maxiter: limit of time and iterations
    """
    W = Winit; H = Hinit; initt = time();
    WtW = dot(W.T,W)
    HHt = dot(H,H.T)
    gradW = (1+lamb)*dot(W, dot(H, H.T)) - dot(V, H.T)
    gradH = (1+lamb)*dot(dot(W.T, W), H) - dot(W.T, V)
    initgrad = norm(r_[gradW, gradH.T])
#     print '     Init gradient norm %f' % initgrad 
    tolW = max(0.001,tol)*initgrad
    tolH = tolW

    for iter in xrange(1,maxiter):
        # stopping condition
        projnorm = norm(r_[gradW[logical_or(gradW<0, W>0)],
                                   gradH[logical_or(gradH<0, H>0)]])
        if projnorm < tol*initgrad or time() - initt > timelimit: break
        (W, gradW, iterW) = nlssubprob_lamb(V.T,H.T,W.T,lamb,tolW,200)
        W = W.T
        gradW = gradW.T
        
        if iterW==1: tolW = 0.1 * tolW

        (H,gradH,iterH) = nlssubprob_lamb(V,W,H,lamb,tolH,200)
        if iterH==1: tolH = 0.1 * tolH
#         if iter % 10 == 0: stdout.write('.')
#     print '     Iter = %d Final proj-grad norm %f' % (iter, projnorm)
    return (W,H,iter,projnorm);

def nlssubprob_lamb(V,W,Hinit,lamb,tol,maxiter):
    """
    H, grad: output solution and gradient
    iter: #iterations used
    V, W: constant matrices
    Hinit: initial solution
    tol: stopping tolerance
    maxiter: limit of iterations
    """

    H = Hinit
    
    WtV = dot(W.T, V)
    WtW = dot(W.T, W) 

    alpha = 1; beta = 0.1;
    for iter in xrange(1, maxiter):  
        grad = (1+lamb)*dot(WtW, H) - WtV
        ## Latex(r"$\nabla^Pf(H)$")
        projgrad = norm(grad[logical_or(grad < 0, H >0)])
        if projgrad < tol: break

        # search step size 
        for inner_iter in xrange(1,20):
            Hn = H - alpha*grad
            Hn = where(Hn > 0, Hn, 0)
            d = Hn-H
            gradd = sum(grad * d)
            dQd = sum(dot(WtW,d) * d)
            # 0.99: 1-0.01
            suff_decr = 0.99*gradd + 0.5*dQd < 0;
            if inner_iter == 1:
                decr_alpha = not suff_decr; Hp = H;
            if decr_alpha: 
                if suff_decr:
                    H = Hn; break;
                else:
                    alpha = alpha * beta;
            else:
                ## Hp==Hn: alpha increased at last iteration
                ## decr_alpha==F and suff_decr==F: alpha不满足继续增大条件
                ## Hp==Hn: 已收敛
                if not suff_decr or (Hp == Hn).all():
                    H = Hp; break;
                else:
                    alpha = alpha/beta; Hp = Hn;
        if iter == maxiter:
            print('     Max iter in nlssubprob')
    return (H, grad, iter)


# # ## LNMF test parameter lamb

# # In[ ]:


# lnmf_aucs=[]
# lambs = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7]
# times = 5
# for lamb in lambs:
#     aucs = []
#     auc = 0
#     for t in range(times):
#         Ws,Hs=sm_lnmf(Gs[1:10],300,lamb)
#         pred = Ws[-1].dot(Hs[-1])
#         G_test = Gs[10]
#         Adj_test = nx.to_pandas_adjacency(G_test, dtype=int, multigraph_weight=sum)
#         Adj_pred = pred
#         edgelist_pos,edgelist_neg = split_data(G_test, Adj_pred)
#         roc_score, ap_score, fpr, tpr, thresholds=get_roc_score(edgelist_pos, edgelist_neg, Adj_pred)
#         aucs.append(roc_score)
#     for a in aucs:
#         auc = auc+a;
#     auc = auc/times;
#     lnmf_aucs.append(auc);
#     print "AUC: ",auc


# # In[ ]:


# ## 保存运算结果
# import pickle
# table = {'lambs': lambs,  
#         'lnmf_aucs': lnmf_aucs}  
# mydb  = open('../temporalData/DBLP/result_graph/20190719_01-10_dim300_lambs', 'w')  
# pickle.dump(table, mydb)

# # ## 读取运算结果
# # mydb  = open('../temporalData/DBLP/result_10-14dim500_iter200_lambs', 'r')  
# # table = pickle.load(mydb)


# # In[ ]:



# ## 作图
# fig = plt.figure()
# l = plt.plot(lambs, lnmf_aucs, label="lNMF")
# x = np.arange(len(lambs))
# plt.legend()
# plt.xlabel("lamba value")
# plt.xticks(lambs,lambs, fontsize=7)
# plt.ylabel("auc")
# plt.title("DBLP1-10 initRandom dim=300")
# ### 添加数据标签
# for x, y in zip(lambs, lnmf_aucs):
#     plt.text(x, y, '%.4f' % y,fontsize=6)
# plt.grid(axis="y")
# fig.dpi=200
# plt.savefig("../temporalData/DBLP/result_graph/20190719_01-10_dim300_lambs_dpi200.eps",dpi=200)
# # plt.show()


# # ## LNMF test parameter iteration

# # In[ ]:


# dim_lnmf_aucs=[]
# dims = [50,100,150,200,250,300,400,500,600,700,800]
# times = 5
# for dim_latent in dims:
#     aucs = []
#     auc = 0
#     for t in range(times):
#         Ws,Hs=sm_lnmf(Gs[1:10],dim_latent,0.2)
    
#         pred = Ws[-1].dot(Hs[-1])
#         G_test = Gs[10]
#         Adj_test = nx.to_pandas_adjacency(G_test, dtype=int, multigraph_weight=sum)
#         Adj_pred = pred
#         edgelist_pos,edgelist_neg = split_data(G_test, Adj_pred)
#         roc_score, ap_score, fpr, tpr, thresholds=get_roc_score(edgelist_pos, edgelist_neg, Adj_pred)
#         aucs.append(roc_score)
#     for a in aucs:
#         auc = auc+a;
#     auc = auc/times;
#     dim_lnmf_aucs.append(auc);
#     print "AUC: ",auc


# # In[ ]:


# ## 保存运算结果
# import pickle
# table = {'dims': dims,  
#         'dim_lnmf_aucs': dim_lnmf_aucs}  
# mydb  = open('../temporalData/DBLP/result_graph/20190719_01-10_lamb02_dims', 'w')  
# pickle.dump(table, mydb)

# # ## 读取运算结果
# # mydb  = open('../temporalData/DBLP/result_10-14dim500_iter200_lambs', 'r')  
# # table = pickle.load(mydb)


# # In[ ]:


# ## 作图
# fig = plt.figure()
# l = plt.plot(dims, dim_lnmf_aucs, label="lNMF")
# x = np.arange(len(lambs))
# plt.legend()
# plt.xlabel("latent dimension")
# plt.xticks(dims,dims, fontsize=7)
# plt.ylabel("auc")
# plt.title("DBLP1-10 initRandom lambda=0.2")
# ### 添加数据标签
# for x, y in zip(dims, dim_lnmf_aucs):
#     plt.text(x, y, '%.4f' % y,fontsize=6)
# plt.grid(axis="y")
# fig.dpi=200
# plt.savefig("../temporalData/DBLP/result_graph/20190719_01-10_lamb02_dims_dpi200.eps",dpi=200)
# # plt.show()

def load_data(filenames, directory):
    # PR DBLP data
    Gs = [];
    ## choose data as train data
    for i in range(0,len(filenames)):
        G = nx.read_edgelist(directory+filenames[i]+".txt", nodetype=int)
        ## remove self loop
        for j in G.nodes():
            if G.has_edge(j,j):
                G.remove_edge(j,j);
        Gs.append(G);
    return Gs;



def nmf_experiment(Gs, lambs, dims, first_G, test_G, times, mydb_path, plt_title, graph_path):
    lnmf_aucs=[]
    print("Graph quickshots from %d to %d"%(first_G,test_G))
    print("repeat experiment times: ",times);
    print("result path: ",mydb_path);
    print("result graph path: ",graph_path);
    for i in range(first_G, test_G):
        print("Graph %d #nodes: "%i,nx.number_of_nodes(Gs[i]));
    print("test Graph #nodes: %d"%nx.number_of_nodes(Gs[test_G]));
    if len(lambs)>1:
        rows_auc=len(lambs)
    if len(dims)>1:
        rows_auc=len(dims)
    Ws=[]
    Hs=[]
    for lamb in lambs:
        for i,dim_latent in enumerate(dims):
            aucs = np.zeros((rows_auc, times));
            aps = np.zeros((rows_auc, times));
            auc = 0
            for t in range(times):
                Ws,Hs=sm_lnmf(Gs[first_G:test_G],dim_latent,lamb,tol=1e-4,init="random")
                pred = Ws[-1].dot(Hs[-1])
                G_test = Gs[test_G]
                Adj_test = nx.to_pandas_adjacency(G_test, dtype=int, multigraph_weight=sum)
                Adj_pred = pred
                edgelist_pos,edgelist_neg = split_data(G_test, Adj_pred)
                roc_score, ap_score, fpr, tpr, thresholds=get_roc_score(edgelist_pos, edgelist_neg, Adj_pred)
                aucs[i,t]=roc_score;
                aps[i,t]=ap_score;
#                 aucs.append(roc_score);
#             for a in aucs:
#                 auc = auc+a;
#             auc = auc/times;
#             lnmf_aucs.append(auc);
            print("AUC: ",aucs[i]);
            print("AP: ",aps[i]);
    table = {'lambs':lambs,
            'dims':dims,
            'aucs':aucs,
            'aps':aps}
    mydb = open(mydb_path, 'w');
    pickle.dump(table, mydb);
    
    
#     if len(lambs)>1:
#         table = {'lambs': lambs,  
#                 'lamb_lnmf_aucs': lnmf_aucs}
#         mydb  = open(mydb_path, 'w')  
#         pickle.dump(table, mydb)
        
#         ## 作图
#         fig = plt.figure()
#         l = plt.plot(lambs, lnmf_aucs, label="lNMF")
#         x = np.arange(len(lambs))
#         plt.legend()
    
#         plt.xlabel("lamb value")
#         plt.xticks(lambs,lambs, fontsize=7)
#         plt.ylabel("auc")
#         plt.title(plt_title)
#         ### 添加数据标签
#         for x, y in zip(lambs, lnmf_aucs):
#             plt.text(x, y, '%.4f' % y,fontsize=6)
#         plt.grid(axis="y")
#         fig.dpi=200
#         plt.savefig(graph_path,dpi=200)
#         # plt.show()
#     if len(dims)>1:
#         table = {'dims': dims,  
#                 'dim_lnmf_aucs': lnmf_aucs}
#         mydb  = open(mydb_path, 'w')  
#         pickle.dump(table, mydb)
        
#         ## 作图
#         fig = plt.figure()
#         l = plt.plot(dims, lnmf_aucs, label="lNMF")
#         x = np.arange(len(dims))
#         plt.legend()
    
#         plt.xlabel("dim_latent")
#         plt.xticks(dims,dims, fontsize=7)
#         plt.ylabel("auc")
#         plt.title(plt_title)
#         ### 添加数据标签
#         for x, y in zip(dims, lnmf_aucs):
#             plt.text(x, y, '%.4f' % y,fontsize=6)
#         plt.grid(axis="y")
#         fig.dpi=200
#         plt.savefig(graph_path,dpi=200)
#         # plt.show()
    
    return aucs;
        
