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
    
    def __init__(self, kernel_params_x, kernel_params_y, components = 1, repetitions = 8, 
                 stopping_criterion = 1e-8, maxit = 800, 
                 kernel_x = 'poly', proj_x = 'l2', Cx = 1, 
                 kernel_y = 'poly', proj_y = 'l2', Cy = 1):
    
    
            
        # This is the constructor method for gradKCCA. It sets all the
        # hyperparameters.
        self.components = components
        self.repetitions = repetitions
        self.stopping_criterion = stopping_criterion
        self.maxit = maxit
        self.kernel_x = kernel_x
        self.kernel_y = kernel_y
        self.proj_x = proj_x
        self.proj_y = proj_y
        self.Cx = Cx
        self.Cy = Cy
        self.kernel_params_x = kernel_params_x
        self.kernel_params_y = kernel_params_y
        
        # Validate the provided parameters
        self._validate_params(kernel_x, kernel_params_x)
        self._validate_params(kernel_y, kernel_params_y)
         
        
    
    def _validate_params(self, kernel_type, params):
        """ Validates if the required parameters for the chosen kernel are provided """
        if kernel_type == "poly":
            if "r" not in params.keys() or "degree" not in params.keys():
                raise ValueError("For polynomial kernel, 'r' and 'degree' must be provided")
        elif kernel_type == "rbf":
            if "sigma" not in params.keys():
                raise ValueError("For RBF kernel, 'sigma' must be provided")
        else:
            raise ValueError("Invalid kernel type. Choose either 'poly' or 'rbf'") 
          
           
    def objective(self, Kx, Ky):
        return Kx.T @ Ky / (np.sqrt(Kx.T @ Kx) * np.sqrt(Ky.T @ Ky))
    
    def kernel_poly(self, x, y, params):
        r = params['r']
        degree = params['degree']
        return (np.dot(x, y) + r) ** degree
    
    def kernel_rbf(self, x, y, params):
        sigma = params['sigma']
        xx = (x * x).sum(-1)  
        yy = np.dot(np.squeeze(y), np.squeeze(y))
        xy = 2 * np.dot(x, np.squeeze(y))  
        return np.exp((-xx + xy - yy) / (2 * sigma ** 2))
    
    def gradient_poly(self, X, u, K, params): 
        r = params['r']
        degree = params['degree']
        t0 = np.dot(X, u) + r
        t1 = np.power(t0, degree)
        t2 = 0.5
        t3 = -t2
        t4 = np.dot(t1.T, t1)
        t5 = np.power(np.dot(K.T, K), t3)
        t6 = np.power(t0, degree - 1)
        t7 = (degree * t5 * np.power(t4, -(1+t2))) / 2
        t8 = np.dot((t1 * t6).T, X)
        gradient = degree * t5 * np.power(t4, t3) * np.dot((K * t6).T, X) - (t7 * np.dot(np.dot(K.T, t1), t8) + t7 * np.dot(np.dot(t1.T, K), t8))
        return gradient
             
        
    def gradient_rbf(self, X, k, params, u):
        
        
        s = params['sigma']
        
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
    
    
    def proj_l1(self, v, b):
        assert b > 0
        if np.linalg.norm(v, 1) < b:
            return v
        u = -np.sort(-np.abs(v))
        sv = np.cumsum(u)
        r = np.where(u > (sv - b) / np.arange(1, u.shape[0]+1))
        if len(r[-1]) > 0:
            # print("not empty")
            rho = r[-1][-1]
            tau = (sv[rho] - b) / (rho + 1)
            theta = np.maximum(0, tau)
            return np.sign(v) * np.maximum(np.abs(v) - theta, 0)
        else:
            # print("empty")
            return v
        
    def proj_l2(self, x, r):
        y = x
        if la.norm(x) > 0.0001:
            y = r * x / la.norm(x)
        return y
    
           
    def fit(self, X, Y):
        
        # This is the training method. Here we learn the model parameters.
        Xm = X
        Ym = Y
        rep_obj = []
        rep_u = np.empty((X.shape[1],self.repetitions))
        rep_v = np.empty((Y.shape[1],self.repetitions))
        comp_u = np.empty((X.shape[1],self.components))
        comp_v = np.empty((Y.shape[1],self.components))
        
        for m in range(self.components): 
            #print('---COMPONENT--- {}'.format(m))
            
            for rep in range(self.repetitions): 
                #print('---REP--- {}'.format(rep))
                
                # Initialize u
                u = np.random.rand(X.shape[1],)
                if self.proj_x == 'l1':
                    u = self.proj_l1(u, self.Cx)
                elif self.proj_x == 'l2':
                    u = self.proj_l2(u, self.Cx) 
                    
                # Initialize v    
                v = np.random.rand(Y.shape[1],)
                if self.proj_y == 'l1':
                    v = self.proj_l1(v, self.Cy)
                elif self.proj_y == 'l2':
                    v = self.proj_l2(v, self.Cy)
                
                # Kernel choice for view X
                if self.kernel_x == 'poly':
                    Ku = self.kernel_poly(Xm, u, self.kernel_params_x)
                elif self.kernel_x == 'rbf':
                    Ku = self.kernel_rbf(Xm, u, self.kernel_params_x)
                
                # Kernel choice for view Y   
                if self.kernel_y == 'poly':
                    Kv = self.kernel_poly(Ym, v, self.kernel_params_y)
                elif self.kernel_y == 'rbf':
                    Kv = self.kernel_rbf(Ym, v, self.kernel_params_y)
                
                diff = 99999
                ite = 0  
                
                # Center Ku and Kv
                cKu = Ku - np.mean(Ku)
                cKv = Kv - np.mean(Kv)
                
                objs = []
                           
                while diff > self.stopping_criterion and ite <= self.maxit:             
                    
                    ite += 1                      
                    obj_old = self.objective(cKu, cKv)                 
                    
                    # Gradient for u
                    if self.kernel_x == 'poly':
                        grad_u = self.gradient_poly(Xm, u, cKv, self.kernel_params_x).T                  
                    elif self.kernel_x == 'rbf':
                        grad_u = self.gradient_rbf(Xm, cKv, self.kernel_params_x, u).T               
                    
                    # Line search for u
                    gamma = la.norm(grad_u, 2)  
                    
                    while True: 
            
                        u_new = u + grad_u * gamma
                        if self.proj_x == 'l1':
                            u_new = self.proj_l1(u_new, self.Cx)
                        elif self.proj_x == 'l2':
                            u_new = self.proj_l2(u_new, self.Cx) 
                            
                        if self.kernel_x == 'poly':
                            Ku_new = self.kernel_poly(Xm, u_new, self.kernel_params_x)
                        elif self.kernel_x == 'rbf':
                            Ku_new = self.kernel_rbf(Xm, u_new, self.kernel_params_x)
                            
                        # Center Ku
                        cKu_new = Ku_new - np.mean(Ku_new)
                        obj_new = self.objective(cKu_new, cKv) 
                        
                        if obj_new > obj_old + 0.0001 * np.abs(obj_old):
                            u = u_new
                            cKu = cKu_new
                            obj = obj_new
                            break
                        else:
                            gamma /= 2
                            if gamma < 1e-13:
                                break
                  
                    obj = obj_new
                    
                    # Gradient for v
                    if self.kernel_y == 'poly':
                        grad_v = self.gradient_poly(Ym, v, cKu, self.kernel_params_y).T 
                    elif self.kernel_y == 'rbf':
                        grad_v = self.gradient_rbf(Ym, cKu, self.kernel_params_y, v).T
                    
                    # Line search for v
                    gamma = la.norm(grad_v, 2)  
                    while True: 
                        v_new = v + grad_v * gamma
                        if self.proj_y == 'l1':
                            v_new = self.proj_l1(v_new, self.Cy)
                        elif self.proj_y == 'l2':
                            v_new = self.proj_l2(v_new, self.Cy)
                        if self.kernel_y == 'poly':   
                            Kv_new = self.kernel_poly(Ym, v_new, self.kernel_params_y)
                        if self.kernel_y == 'rbf':   
                            Kv_new = self.kernel_rbf(Ym, v_new, self.kernel_params_y)
                            
                            
                        # Center Kv
                        cKv_new = Kv_new - np.mean(Kv_new)
                            
                        obj_new = self.objective(cKu, cKv_new)      
                        if obj_new > obj_old + 0.0001 * np.abs(obj_old):
                            v = v_new
                            cKv = cKv_new
                            obj = obj_new
                            break
                        else:
                            gamma /= 2
                            if gamma < 1e-13:
                                break 
                            
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
        # kernel choice for view x
        if self.kernel_x == 'poly':
            Kx = self.kernel_poly(X, self.u_, self.kernel_params_x)
        elif self.kernel_x == 'rbf':
            Kx = self.kernel_rbf(X, self.u_, self.kernel_params_x)
            
        if self.kernel_y == 'poly':
            Ky = self.kernel_poly(Y, self.v_, self.kernel_params_y)
        elif self.kernel_y == 'rbf':
            Ky = self.kernel_rbf(Y, self.v_, self.kernel_params_y)    
        
        cKx = Kx - np.mean(Kx)
        cKy = Ky - np.mean(Ky)
            
        return self.objective(cKx,cKy).item()
 
    
  
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
   
    model = GradKCCA(kernel_x = 'poly', kernel_params_x = {'r':0, 'degree':2},
                       kernel_y = 'poly', kernel_params_y = {'r':0, 'degree':2},
                       proj_x = 'l1', proj_y = 'l1').fit(Xtrain,Ytrain)
    print(model.u_)
    print(model.v_)
    print('Training Correlation: {:.3f}'.format(model.value_))
    test_corr = model.predict(Xtest,Ytest)
    print('Test Correlation: {:.3f}'.format(test_corr))
    
    plt.plot(X @ model.u_, Y @ model.v_,'bo')
    plt.show()
   
      
if __name__ == "__main__":
    main()
    
    

        
    
    




