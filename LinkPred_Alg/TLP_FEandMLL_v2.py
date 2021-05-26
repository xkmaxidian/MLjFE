from nmf import *
from sklearn.decomposition import TruncatedSVD
import numpy.linalg as LA
import numpy as np;
import networkx as nx;
import pandas as pd;
import gc;
from numpy.linalg import multi_dot

def DeepWalk_matrix(A, t):
    M = A;
    for i in range(1,t):
        M = M.dot(A)+A;
    M = np.log(M/t,where=(M!=0))
    return M;


## parameters:
### k:dim_latent
### t:deep walk windows length
### alpha:feature extraction weight
### beta:temporal smoothness weight
### larg:largrange multipleier
### max_time:algorithm reach max_time then stop
### max_iter:algorithm maximun iteration
### tol:stop criterian for projnorm

def TLP_FEandMLL_v2(Gs, dim_latent, t, alpha, beta, larg, max_time, max_iter, tol=1e-5):
    gc.enable();
    Xs=[]
    Zs=[]
    Ys=[]
    for i in range(len(Gs)):
#         print "nodes: ", nx.nYmber_of_nodes(Gs[i])
        A = nx.to_pandas_adjacency(Gs[i],dtype=np.float64)
#         ## initialed by TSVD
#         svd = TrYncatedSVD(n_components=dim_latent, n_iter=50)
#         Y = svd.fit_transform(A.values)
#         V = svd.components_
        M = DeepWalk_matrix(A,t);
        
        X = np.random.rand(len(A),dim_latent);
        Y = np.random.rand(dim_latent,len(A));
        Z = np.random.rand(dim_latent,len(A));
        for idx in range(len(A)):
            X[idx,:] = X[idx,:]/LA.norm(X[idx,:])
            Z[:,idx] = Z[:,idx]/LA.norm(Z[:,idx])
            Y[:,idx] = Y[:,idx]/LA.norm(Y[:,idx])
        X = pd.DataFrame(X, index=A.index);
        Y = pd.DataFrame(Y, columns=A.columns);
        Z = pd.DataFrame(Z, columns=A.columns);
        if i==0:
            X,Z,Y = FEandMLL_i0(M,A,X,Z,Y,dim_latent,alpha,larg,max_time,max_iter,tol);
        else:
            X,Z,Y = FEandMLL(M,A,L,X,Z,Zp,Y,dim_latent,alpha,beta,larg,max_time,max_iter,tol);
        ## L and Zp is with previoYs time
        L = nx.normalized_laplacian_matrix(Gs[i], nodelist=A.index).toarray()
        L = pd.DataFrame(L,index=A.index,columns=A.columns)
#         print "L:",np.shape(L)
        Zp=pd.DataFrame(Z,columns=A.columns);
        
        Xs.append(pd.DataFrame(X, index=A.index))
        Zs.append(pd.DataFrame(Z, columns=A.columns))
        Ys.append(Y)
        gc.collect();
    return Xs,Zs,Ys;





def FEandMLL_i0(M, A, init_X, init_Z, init_Y, dim_latent, alpha, larg, max_time, max_iter, tol):
    ### bet: the searching rate
    bet = 0.1; sigma = 0.01;
    X=init_X; Z=init_Z; Y=init_Y;I=np.eye(dim_latent);initt = time();timelimit=max_time;
    gradX = -(A.T.dot(Z.T)+alpha*M.dot(Y.T))+X.dot((Z.dot(Z.T)+alpha*Y.dot(Y.T)+larg*I));
    gradZ = -X.T.dot(A.T)+X.T.dot(X).dot(Z);
#     initgrad = norm(r_[gradX, gradY])
    initgradX = LA.norm(gradX)
    initgradZ = LA.norm(gradZ)
    
    initgrad = initgradX+initgradZ;
    print("initgrad: ",initgrad);
    tol_X = max(0.001, tol)*initgradX;
    tol_Z = max(0.001, tol)*initgradZ;
    
    for iter in range(1,max_iter):
#         projnorm = norm(r_[gradX[logical_or(gradX<0, X>0)],
#                                    gradY[logical_or(gradY<0, Z>0)]])
        projnorm = LA.norm(gradX[logical_or(gradX<0, X>0)])+LA.norm(gradZ[logical_or(gradZ<0, Z>0)])
        if projnorm < tol*initgrad or time() - initt > timelimit: 
            break
        X,gradX,iterX = argminX(M, A, X, Z, Y, alpha, larg, bet, sigma, 1000, tol_X);
        X = pd.DataFrame(X, index=A.index);
        Z,gradZ,iterZ = argminZ_i0(M, A, X, Z, Y, alpha, bet, sigma, 1000, tol_Z);
        Z = pd.DataFrame(Z, columns=A.columns);
        Y = argminY(M, X);
        Y = pd.DataFrame(Y, columns=A.columns);
    print("projnorm: ",projnorm);
    return X,Z,Y

## Zp is DataFrame stYctYre
def FEandMLL(M, A, L, init_X, init_Z, Zp, init_Y, dim_latent, alpha, beta, larg, max_time, max_iter, tol):
    ### bet: the searching rate
    bet = 0.1; sigma = 0.01;
    X=init_X; Z=init_Z; Y=init_Y;I=np.eye(dim_latent);initt = time();timelimit=max_time;
#     gradX = alpha*( np.dot(np.dot(X,Z.T),Z)-np.dot(M,Z) ) + ( np.dot(np.dot(np.dot(np.dot(X,Z),Y.T),Y),Z.T)-np.dot(np.dot(A,Y),Z) );
#     gradY = alpha*(np.dot(np.dot(Z,X.T),X)-np.dot(M.T,X))+(np.dot(np.dot(np.dot(np.dot(Z,Y.T),X.T),X),Y)-np.dot(np.dot(A.T,X),Y));
    gradX = -(A.T.dot(Z.T)+alpha*M.dot(Y.T))+X.dot((Z.dot(Z.T)+alpha*Y.dot(Y.T)+larg*I));
    gradZ = -X.T.dot(A.T)+X.T.dot(X).dot(Z)+beta*Z;
    gradZ = pd.DataFrame(gradZ, columns=Z.columns);
    
    drop = []
    for idx in Zp.columns:
        if idx not in A.columns:
            drop.append(idx);
    gradZ = gradZ.sub(beta*Zp,fill_value=0);
    gradZ = gradZ.drop(columns=drop);
    
#     initgrad = norm(r_[gradX, gradY])
    initgradX = LA.norm(gradX)
    initgradZ = LA.norm(gradZ)
    initgrad = initgradX+initgradZ;
    print("initgrad: ",initgrad)
    tol_X = max(0.001, tol)*initgradX;
    tol_Z = max(0.001, tol)*initgradZ;
    
    for iter in range(1,max_iter):
#         projnorm = norm(r_[gradX[logical_or(gradX<0, X>0)],
#                                    gradY[logical_or(gradY<0, Z>0)]])
        projnorm = LA.norm(np.nan_to_num(gradX[logical_or(gradX<0, X>0)]))+LA.norm(np.nan_to_num(gradZ[logical_or(gradZ<0, Z>0)]))
        if projnorm < tol*initgrad or time() - initt > timelimit: 
            break
        X,gradX,iterX = argminX(M, A, X, Z, Y, alpha, larg, bet, sigma, 1000, tol_X);
        X = pd.DataFrame(X, index=A.index);
        Z,gradZ,iterZ = argminZ(M, A, L, X, Z, Zp, Y, alpha, beta, bet, sigma, 1000, tol_Z);
        Z = pd.DataFrame(Z, columns=A.columns);
        Y = argminY(M, X);
        Y = pd.DataFrame(Y, columns=A.columns);
    print("projnorm: ",projnorm);
    return X,Z,Y

## argminX, argminZ, argminZ_io, argminY all return result with numpy.array type
def argminX(M, A, init_X, Z, Y, alpha, larg, bet, sigma, max_iter, tol):
    X = init_X.values;
    A = A.values;
    M = M.values;
    Z = Z.values;
#     X = init_X;
#     MY = np.dot(M,Z);
#     XYtY = np.dot(np.dot(X,Z.T),Z)
#     AYY = np.dot(np.dot(A,Y),Z)
#     XYYtYYt = np.dot(np.dot(np.dot(np.dot(X,Z),Y.T),Y),Z.T);
    I = np.eye(np.shape(Z)[0]);
    YYt = np.dot(Y,Y.T);
    ZZt = np.dot(Z,Z.T)
    WtZt = np.dot(A.T,Z.T)
    MYt = np.dot(M,Y.T)
    bet=bet;sigma=sigma;
    for iter in range(1,max_iter):
        grad = -(WtZt+alpha*MYt)+X.dot((ZZt+alpha*YYt+larg*I));
        ### stepsize t
        t=1;
        projgrad = LA.norm(grad[logical_or(grad < 0, X >0)])
        if projgrad < tol: 
            break
        # search step size 
        for inner_iter in range(1,20):
            Xn = X-t*grad;
            d = (Xn-X);
            gradd = np.sum((grad*d))
            dQd = np.sum((np.dot(d,(ZZt+alpha*YYt+larg*I))*d));
            suff_decr = ((1-sigma)*gradd+0.5*dQd) < 0;
            if inner_iter == 1:
                decr_t = not suff_decr; 
                Xp = X;
            if decr_t: 
                if suff_decr:
                    X = Xn; break;
                else:
                    t = t * bet;
            else:
                if not suff_decr or np.allclose(Xp,Xn,equal_nan=True):
                    X = Xp; break;
                else:
                    t = t/bet; Xp = Xn;
        if iter == max_iter:
            print('     Max iter in nlssubprob')
    #X[X<0]=0;
    return X,grad,iter;

def argminZ_i0(M, A, X, init_Z, Y, alpha, bet, sigma, max_iter, tol):
    X = X.values;
    A = A.values;
    M = M.values;
    Z = init_Z.values;
    XtX = np.dot(X.T,X);
    XtWt = np.dot(X.T,A.T);
    I = np.eye(np.shape(Z)[0])
    bet=bet;sigma=sigma;
    
    
    for iter in range(1,max_iter):
        grad = -XtWt+np.dot(XtX,Z);
        t=1;
        projgrad = LA.norm(grad[logical_or(grad < 0, Z >0)])
        if projgrad < tol: 
            break
            
        # search step size 
        for inner_iter in range(1,20):
            Zn = Z-t*grad;
            d = Zn-Z;
            gradd = np.sum(grad*d)
            dQd = np.sum(d*np.dot(XtX,d))
            suff_decr = (1-sigma)*gradd+0.5*dQd<0;
            if inner_iter == 1:
                decr_t = not suff_decr; Zp = Z;
            if decr_t: 
                if suff_decr:
                    Z = Zn; break;
                else:
                    t = t * bet;
            else:
                if not suff_decr or np.allclose(Zp,Zn,equal_nan=True):
                    Z = Zp; break;
                else:
                    t = t/bet; Zp = Zn;
        if iter == max_iter:
            print('     Max iter in nlssYbprob')
    #Z[Z<0]=0;
    return Z,grad,iter;

# Zp is DataFrame structure
def argminZ(M, A, L, X, init_Z, Zp, Y, alpha, beta, bet, sigma, max_iter, tol):
    Z = init_Z;
    XtX = np.dot(X.T,X);
    XtWt = np.dot(X.T,A.T)
    I = np.eye(np.shape(Z)[0])
    bet=bet;sigma=sigma;
    drop = []
    for idx in Zp.columns:
        if idx not in A.columns:
            drop.append(idx);
    Zp=Zp.drop(columns=drop);
    for iter in range(1,max_iter):
        grad = -XtWt+np.dot(XtX,Z)+beta*(Z.sub(Zp,fill_value=0));
        grad = grad.values;
        #grad = pd.DataFrame(grad, columns=Z.columns);
        #grad = grad.sub(beta*Zp,fill_value=0);
        #grad = grad.drop(columns=drop);
        if np.any(np.isnan(grad)):
            print("grad has none value")
        t=1;
        projgrad = LA.norm(grad[logical_or(grad < 0, Z >0)])
        if projgrad < tol: 
            break
            
        # search step size 
        for inner_iter in range(1,20):
            Zn = Z-t*grad;
            if np.any(np.isnan(Zn)):
                print("Zn has none value")
            d = Zn.values-Z.values;
            gradd = np.sum(np.sum(grad*d))
            dQd = np.sum(d*np.dot(XtX+beta*I,d))
            suff_decr = (1-sigma)*gradd+0.5*dQd < 0;
            if np.any(Zn.isna()):
                print("Zn has none value in argminZ")
            if inner_iter == 1:
                decr_t = not suff_decr; Zp = Z;
            if decr_t: 
                if suff_decr:
                    Z = Zn; 
                    break;
                else:
                    t = t * bet;
            else:
                if not suff_decr or np.allclose(Zp,Zn,equal_nan=True):
                    Z = Zp; 
                    break;
                else:
                    t = t/bet; Zp = Zn;
        if iter == max_iter:
            print('     Max iter in nlssYbprob')
    #Z[Z<0]=0;
    return Z,grad,iter;

def argminY(M, X):
    Xi=LA.pinv(X)
    Y = np.dot(Xi,M)
    return Y;
