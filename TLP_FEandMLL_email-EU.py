import NMFexperiment;
import numpy as np;
import numpy.linalg as LA
import networkx as nx;
import pandas as pd;
import pickle;
from TLP_FEandMLL_v2 import *;
from NMFexperiment import split_data;
from NMFexperiment import get_roc_score;
from NMFexperiment import AUC_AP_Score;
import gc;

## email-EU dataset
# directory = "/home/tanshiyin/Documents/machineLearning/dynamicNetwork/temporalData/sx-askubuntu/sx-askubuntu_ByMonth/"
directory = "D:\\machineLearning/dynamicNetwork/temporalData/email-EU-core/email-EU-core-temporal-ByMonth/"
filenames=["1970-01.txt","1970-02.txt","1970-03.txt","1970-04.txt","1970-05.txt","1970-06.txt","1970-07.txt", \
          "1970-08.txt","1970-09.txt","1970-10.txt","1970-11.txt","1970-12.txt"]
Gs = [];
## choose data as train data
for i in range(0,8):
    G = nx.read_edgelist(directory+filenames[i], nodetype=int)
    ## remove self loop
    for j in G.nodes():
        if G.has_edge(j,j):
            G.remove_edge(j,j);
    Gs.append(G);

times=2
mydb_path="D:\\machineLearning/dynamicNetwork/temporalData/email-EU-core/TLP_FEandMLL/result/Gs197001-197008_dim400_a_0-100_b1times2_tol1e-2"
dims = [100,200,300,400,500];
alphas = np.arange(0,2,0.1);
betas = np.arange(0,2,0.1);
larg=0.5

for G in Gs:
    print "nodes number:",nx.number_of_nodes(G);

aucs = np.zeros((len(dims),len(alphas),len(betas),times));
aps = np.zeros((len(dims),len(alphas),len(betas),times));
errors = np.zeros((len(dims),len(alphas),len(betas),times));
for i,dim in enumerate(dims):
    for j,alpha in enumerate(alphas):
        for k,beta in enumerate(betas):
            print "dim, alpha, beta: ",dim,alpha,beta;
            for t in range(times):
                Xs,Zs,Ys = TLP_FEandMLL_v2(Gs[0:7], dim, 3, alpha, beta, larg, 600, 200, tol=1e-2)
#                 pred = Xs[-1].dot(Us[-1]).dot(Ys[-1].T)
                pred = pd.DataFrame()
                for l in range(len(Xs)):
                    pred = pred.add(Xs[l].dot(Zs[l]), fill_value=0)
                    pred=pred.fillna(0)
                G_test = Gs[7]
                Adj_test = nx.to_pandas_adjacency(G_test, dtype=int)
                Adj_pred = pred
                edgelist_pos,edgelist_neg = split_data(G_test, Adj_pred)
                roc_score, ap_score, fpr, tpr, thresholds=get_roc_score(edgelist_pos, edgelist_neg, Adj_pred)
                #error = LA.norm(Adj_pred.sub(Adj_test,fill_value=0));
                #print "error: ",error;
                #errors[i,j,k,t]=error;
                print "AUC:  ",roc_score
                print "ap_score: ",ap_score
                aucs[i,j,k,t]=roc_score;
                aps[i,j,k,t]=ap_score;
                gc.collect();

table = {'dims':dims,
         'alphas':alphas,
         'betas':betas,
         'aucs':aucs,
        'aps':aps}

mydb = open(mydb_path, 'w');
pickle.dump(table, mydb);