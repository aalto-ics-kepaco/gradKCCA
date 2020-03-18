#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 09:12:21 2019

@author: vuurtio
"""

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
np.seterr(divide='ignore', invalid='ignore')


class GradKCCA:
    
    def __init__(self, components = 1, repetitions = 8, 
                 stopping_criterion = 1e-8, maxit = 800, 
                 kernel_x = 'poly', kernel_x_param = 1, proj_x = 'l2',
                 kernel_y = 'poly', kernel_y_param = 1, proj_y = 'l2'):
        # This is the constructor method for gradKCCA. It sets all the
        # hyperparameters.
        self.components = components
        self.repetitions = repetitions
        self.stopping_criterion = stopping_criterion
        self.maxit = maxit
        self.kernel_x = kernel_x
        self.kernel_x_param = kernel_x_param
        self.kernel_y = kernel_y
        self.kernel_y_param = kernel_y_param
        self.proj_x = proj_x
        self.proj_y = proj_y
           
    def objective(self, Kx, Ky):
        return Kx.T @ Ky / (np.sqrt(Kx.T @ Kx) * np.sqrt(Ky.T @ Ky))
    
    def kernel_poly(self, W, t, d):
            return (W @ t)**d
        
    def kernel_rbf(self, W, t, sigma):
        xx = np.sum(np.abs(W)**2, axis=-1)**(1./2)
        yy = la.norm(t)
        xy = 2 * W @ t
        return np.exp((-xx + xy - yy) / (2 * sigma**2))
             
    def gradient_poly(self,X,u,r,d,K):      
        t0 = r + X @ u;
        t1  = np.power(t0,d);
        t2 = 0.5;
        t3 = -t2;
        t4 = t1.T @ t1;
        t5 = np.power((K.T @ K),t3);
        t6 = np.power(t0,(d-1));
        t7 = (d * t5 * t4**(-(1+t2)))/2;
        t8 = (t1 * t6).T @ X;
        return d * t5 * t4**t3 * (K * t6).T @ X - (t7 * K.T @ t1 * t8 + t7 * t1.T @ K * t8)
    
    def gradient_rbf(self, X, k, s, u):
        
        x = np.sum(np.abs(X)**2, axis=-1)**(1./2)
        
        assert isinstance(X, np.ndarray)
        dim = X.shape
        assert len(dim) == 2
        X_rows = dim[0]
        X_cols = dim[1]
        assert isinstance(k, np.ndarray)
        dim = k.shape
        assert len(dim) == 1
        k_rows = dim[0]
        if isinstance(s, np.ndarray):
            dim = s.shape
            assert dim == (1, )
        assert isinstance(u, np.ndarray)
        dim = u.shape
        assert len(dim) == 1
        u_rows = dim[0]
        assert isinstance(x, np.ndarray)
        dim = x.shape
        assert len(dim) == 1
        x_rows = dim[0]
        assert u_rows == X_cols
        assert k_rows == x_rows == X_rows
    
        t_0 = np.exp((((X).dot(u) - x) - ((la.norm(u) ** 2) * np.ones(k_rows))))
        t_1 = (1 / 2)
        t_2 = -t_1
        t_3 = (4 * (s ** 2))
        t_4 = (t_0).dot(t_0)
        t_5 = (1 + t_1)
        t_6 = ((k).dot(k) ** t_2)
        t_7 = (t_0).dot(k)
        t_8 = (t_4 ** -t_5)
        t_9 = (16 ** t_5)
        t_10 = (s ** (4 * t_5))
        t_11 = (s ** 6)
        t_12 = (t_0 * t_0)
        t_13 = ((t_8 * ((t_9 * t_10) * t_6)) / ((8 * t_11) * 16))
        t_14 = (X.T).dot(t_12)
        t_15 = ((((t_8 * t_9) * (t_10 * t_6)) / ((4 * t_11) * 16)) * np.sum(t_12))
        t_16 = (k).dot(t_0)
        t_17 = (((t_4 ** t_2) * (16 ** t_1)) * ((s ** (4 / 2)) * t_6))
        t_18 = (k * t_0)
        #functionValue = (((((t_4 / (t_3 ** 2)) ** t_2) * t_6) * t_7) / t_3)
        gradient = ((((((t_15 * (t_7 * u)) - (t_13 * (t_7 * t_14))) - (t_13 * (t_16 * t_14))) + (t_15 * (t_16 * u))) + ((t_17 / t_3) * (X.T).dot(t_18))) - ((((2 * t_17) / t_3) * np.sum(t_18)) * u))  
        return gradient
    
     
    def backracking_line_search(self, w, gw, stp, X, K, obj_old): 
        while True:                      
            w_new = w + gw * stp
            w_new = w_new / la.norm(w_new)
            Kw_new = X @ w_new
            obj_new = self.objective(Kw_new,K)                       
            if obj_new > obj_old + 0.0001 * np.abs(obj_old):
                w = w_new
                Kw = Kw_new
                obj = obj_new
                #print('new objective better')
                break
            elif stp < 1e-20:
                w = w_new
                Kw = Kw_new
                obj = obj_new
                #print('step size reached')
                break
            else:
                stp /= 2
                
        return w, Kw, obj
    
    def proj_l1(self, v, b): 
        assert b > 0
        if la.norm(v,1) < b:
            return v     
        u = -np.sort(-np.abs(v))
        sv = np.cumsum(u)
        r = np.where(u > (sv - b) / np.arange(1,u.shape[0]+1))
        if len(r[-1]) > 0:
            #print("not empty")
            rho = r[-1][-1]
            tau = (sv[rho] - b) / rho
            theta = np.maximum(0, tau)
            return np.sign(v) * np.maximum(np.abs(v) - theta, 0)
        else:
            #print("empty")
            return v
        
           
    def fit(self, X, Y):
        # This is the training method. Here we learn the model parameters.
        Xm = X
        Ym = Y
        objs = []
        rep_obj = []
        rep_u = np.empty((X.shape[1],self.repetitions))
        rep_v = np.empty((Y.shape[1],self.repetitions))
        comp_u = np.empty((X.shape[1],self.components))
        comp_v = np.empty((Y.shape[1],self.components))
        
        for m in range(self.components): 
            #print('---COMPONENT--- {}'.format(m))
            
            for rep in range(self.repetitions): 
                #print('---REP--- {}'.format(rep))
                u = np.random.rand(X.shape[1],)
                if self.proj_x == 'l1':
                    u = self.proj_l1(u, 1)
                else:
                    u = u / la.norm(u)
                v = np.random.rand(Y.shape[1],)
                if self.proj_y == 'l1':
                    v = self.proj_l1(v, 1)
                else:
                    v = v / la.norm(v)
                
                # kernel choice for view x
                if self.kernel_x == 'poly':
                    Ku = self.kernel_poly(Xm, u, self.kernel_x_param)
                elif self.kernel_x == 'rbf':
                    Ku = self.kernel_rbf(Xm, u, self.kernel_x_param)
                
                # kernel choice for view y   
                if self.kernel_y == 'poly':
                    Kv = self.kernel_poly(Ym, v, self.kernel_y_param)
                elif self.kernel_y == 'rbf':
                    Kv = self.kernel_rbf(Ym, v, self.kernel_y_param)
                
                diff = 99999
                ite = 0               
                while diff > self.stopping_criterion and ite <= self.maxit:                  
                    ite += 1                                      
                    obj_old = self.objective(Ku,Kv) 
                    if self.kernel_x == 'poly':
                        grad_u = self.gradient_poly(Xm,u,0,self.kernel_x_param,Kv).T                       
                    elif self.kernel_x == 'rbf':
                        grad_u = self.gradient_rbf(Xm,Kv,self.kernel_x_param,u).T
                    gamma = la.norm(grad_u)    
                    u, Ku, obj_new = self.backracking_line_search(u,grad_u,gamma,Xm,Kv,obj_old)
                    if self.proj_x == 'l1':
                        u = self.proj_l1(u, 1)
                    else:
                        u = u / la.norm(u)
                    obj = obj_new
                    objs.append(obj)
                    if self.kernel_y == 'poly':
                        grad_v = self.gradient_poly(Ym,v,0,self.kernel_y_param,Ku).T 
                    elif self.kernel_y == 'rbf':
                        grad_v = self.gradient_rbf(Ym,Ku,self.kernel_y_param,v).T
                    gamma = la.norm(grad_v) 
                    #print('line search for v')
                    v, Kv, obj_new = self.backracking_line_search(v,grad_v,gamma,Ym,Ku,obj_new)  
                    if self.proj_y == 'l1':
                        v = self.proj_l1(v, 1)
                    else:
                        v = v / la.norm(v)
                    obj = obj_new
                    objs.append(obj)  
                    diff = np.abs(obj-obj_old)/np.abs(obj+obj_old);  
                    #print('Training Objective: {:.3f}, Iter: {}'.format(obj,ite))
                print('Comp: {}, Rep: {}, Stop: {:.3f}, Iter: {}'.format(m,rep,obj,ite))    
                rep_u[:,rep] = u
                rep_v[:,rep] = v  
                rep_obj.append(obj)
                
            max_obj = max(rep_obj)
            idx_obj = rep_obj.index(max_obj)
            comp_u[:,m] = rep_u[:,idx_obj]
            comp_v[:,m] = rep_v[:,idx_obj]
            
            # different deflation strategies here
            if self.components > 1:
                Xm = Xm - (np.outer(comp_u[:,m], comp_u[:,m]) @ Xm.T).T
                Ym = Ym - (np.outer(comp_v[:,m], comp_v[:,m]) @ Ym.T).T
                      
        # attributes learnt
        self.u_ = comp_u
        self.v_ = comp_v
        self.value_ = max_obj
        
        return self
                  
    def predict(self, X, Y):
        # method for computing test correlation
        Kx = X @ self.u_
        Ky = Y @ self.v_
        return self.objective(Kx,Ky)
 
    
  
def generate_data(n,p,q):
    X = np.random.uniform(-1,1,[n,p])
    Y = np.random.uniform(-1,1,[n,q])
    Y[:,2] = X[:,2] + X[:,3] - Y[:,3] + np.random.normal(0,0.05,n)
    #Y[:,2] = np.power(X[:,2] + X[:,3],3) - Y[:,3] + np.random.normal(0,0.05,n)
    #Y[:,4] = np.exp(X[:,4] + X[:,5]) - Y[:,5] + np.random.normal(0,0.05,n)
    return X, Y
    
def normalise_data(X,Y):   
    Xn = (X - np.mean(X)) / np.std(X)
    Yn = (Y - np.mean(Y)) / np.std(Y)   
    return Xn, Yn

def partition_data(X,Y):  
    train = int(round(2 * X.shape[0] / 3)) 
    indices = np.random.permutation(X.shape[0])
    training_idx, test_idx = indices[:train], indices[train:]
    Xtrain, Xtest, Ytrain, Ytest = X[training_idx,:], X[test_idx,:], Y[training_idx,:], Y[test_idx,:]     
    return Xtrain, Xtest, Ytrain, Ytest
    
def main():
    np.set_printoptions(precision=2)   
     
    X, Y = generate_data(1000,8,8)   
    Xn, Yn = normalise_data(X,Y)
    Xtrain, Xtest, Ytrain, Ytest = partition_data(Xn,Yn) 
   
    kernel1 = GradKCCA(kernel_x = 'poly', kernel_y = 'poly',
                       proj_x = 'l1', proj_y = 'l1',
                       kernel_x_param = 1, kernel_y_param = 1).fit(Xtrain,Ytrain)
    print(kernel1.u_)
    print(kernel1.v_)
    print('Training Correlation: {:.3f}'.format(kernel1.value_))
    test_corr = kernel1.predict(Xtest,Ytest)
    print('Test Correlation: {:.3f}'.format(test_corr[0,0]))
    
    plt.plot(X @ kernel1.u_, Y @ kernel1.v_,'bo')
    plt.show()
   
      
if __name__ == "__main__":
    main()
    
    

        
    
    




