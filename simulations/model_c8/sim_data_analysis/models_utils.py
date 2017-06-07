# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 18:23:57 2015

@author: asifzubair
"""

import numpy as np

def tridiag(a, b, c, k1=-1, k2=0, k3=1):
    return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)
    

def omega(which,Hb,B,C,T,params):

    #1-Hunchback, 2-Knirps, 3-Kruppel, 4-Giant
    alpha, K, K1, K2, K3, Co, Ns = params
    
    if which == 1:
        f = p(B, K3, Co, Ns)*p(Hb, K, Co, Ns)
    elif which == 2:
        f = p(B, K3, Co, Ns)*(1-p(Hb, K1, Co, Ns))
    elif which == 3:
        f = p(Hb, K, Co, Ns)*(1-p(Hb, K, Co, Ns))
    elif which == 4:
        f = (1-((1-p(B, K, Co, Ns))*(1-p(C, K, Co, Ns))))
        
    return f

def omega_B_Kr(which,Hb,B,C,T,params):

    #1-Hunchback, 2-Knirps, 3-Kruppel, 4-Giant
    alpha, K, K1, K2, K3, K4, Co, Ns = params
    
    if which == 1:
        f = p(B, K3, Co, Ns)*p(Hb, K, Co, Ns)
    elif which == 2:
        f = p(B, K3, Co, Ns)*(1-p(Hb, K1, Co, Ns))
    elif which == 3:
        f = p(B, K4, Co, Ns)*p(Hb, K, Co, Ns)*(1-p(Hb, K, Co, Ns))
    elif which == 4:
        f = (1-((1-p(B, K3, Co, Ns))*(1-p(C, K, Co, Ns))))
        
    return f

        
def p(u, K, Co, Ns):
    return (1 - ((1.*Co)/(Co + (1+Co*K*u)**Ns - 1)))


def G(which, V1,V2,V3,V4, x, Bi, Ci, Ti, params, omegas):
    
    #1-Hunchback, 2-Knirps, 3-Kruppel, 4-Giant
    alpha, K, K1, K2, K3, Co, Ns = params  
    omega1,omega2,omega3,omega4 = omegas
    
    Hb = V1
    Kni = V2
    Kr = V3
    Gt = V4    
    
    if which == 1:
        f = omega1*alpha*p(Bi[x], K3, Co, Ns)*p(Hb[x], K, Co, Ns)*(1-p(Kni[x], K, Co, Ns))
    elif which == 2:
        f = omega2*alpha*p(Bi[x], K3, Co, Ns)*(1-p(Hb[x], K1, Co, Ns))*(1-p(Ti[x], K, Co, Ns))
    elif which == 3:
        f = omega3*alpha*p(Hb[x], K, Co, Ns)*(1-p(Hb[x], K, Co, Ns))*(1-p(Gt[x], K, Co, Ns))
    elif which == 4:
        f = omega4*alpha*(1-(1-p(Bi[x], K3, Co, Ns))*(1-p(Ci[x], K, Co, Ns)))*(1-p(Kr[x], K2, Co, Ns))*(1-p(Ti[x], K, Co, Ns))
    
    return f


def G_B_Kr(which, V1,V2,V3,V4, x, Bi, Ci, Ti, params, omegas):
    
    #1-Hunchback, 2-Knirps, 3-Kruppel, 4-Giant
    alpha, K, K1, K2, K3, K4, Co, Ns = params
    omega1,omega2,omega3,omega4 = omegas
    
    Hb = V1
    Kni = V2
    Kr = V3
    Gt = V4    
    
    if which == 1:
        f = omega1*alpha*p(Bi[x], K3, Co, Ns)*p(Hb[x], K, Co, Ns)*(1-p(Kni[x], K, Co, Ns))
    elif which == 2:
        f = omega2*alpha*p(Bi[x], K3, Co, Ns)*(1-p(Hb[x], K1, Co, Ns))*(1-p(Ti[x], K, Co, Ns))
    elif which == 3:
        f = omega3*alpha*p(Bi[x], K4, Co, Ns)*p(Hb[x], K, Co, Ns)*(1-p(Hb[x], K, Co, Ns))*(1-p(Gt[x], K, Co, Ns))
    elif which == 4:
        f = omega4*alpha*(1-(1-p(Bi[x], K3, Co, Ns))*(1-p(Ci[x], K, Co, Ns)))*(1-p(Kr[x], K2, Co, Ns))*(1-p(Ti[x], K, Co, Ns))
    
    return f


def proj(which, n, V1,V2,V3,V4, B, C, T, params, omegas):
    
    #Fn = np.zeros(n+1)
    x = range(n+1)
    
    if which == 1:
        return G(1, V1,V2,V3,V4, x, B, C, T, params, omegas)
    elif which == 2:
        return G(2, V1,V2,V3,V4, x, B, C, T, params, omegas)
    elif which == 3:
        return G(3, V1,V2,V3,V4, x, B, C, T, params, omegas)
    elif which == 4:
        return G(4, V1,V2,V3,V4, x, B, C, T, params, omegas)


def proj_B_Kr(which, n, V1,V2,V3,V4, B, C, T, params, omegas):
    
    #Fn = np.zeros(n+1)
    x = range(n+1)
    
    if which == 1:
        return G_B_Kr(1, V1,V2,V3,V4, x, B, C, T, params, omegas)
    elif which == 2:
        return G_B_Kr(2, V1,V2,V3,V4, x, B, C, T, params, omegas)
    elif which == 3:
        return G_B_Kr(3, V1,V2,V3,V4, x, B, C, T, params, omegas)
    elif which == 4:
        return G_B_Kr(4, V1,V2,V3,V4, x, B, C, T, params, omegas)

    
def nonlinear(n,U,B,C,T,params,omegas):

    k = range(0,4*n+4,n+1)
    
    V1 = U[k[0]:n+1+k[0]]
    V2 = U[k[1]:n+1+k[1]]
    V3 = U[k[2]:n+1+k[2]]
    V4 = U[k[3]:n+1+k[3]]
    
    fn1 = proj(1,n,V1,V2,V3,V4,B,C,T, params, omegas)
    fn2 = proj(2,n,V1,V2,V3,V4,B,C,T, params, omegas)
    fn3 = proj(3,n,V1,V2,V3,V4,B,C,T, params, omegas)
    fn4 = proj(4,n,V1,V2,V3,V4,B,C,T, params, omegas)

    #Fn= inv(MN)*[fn1;fn2;fn3;fn4];

    return np.concatenate([fn1,fn2,fn3,fn4])
    
def nonlinear_B_Kr(n,U,B,C,T,params,omegas):

    k = range(0,4*n+4,n+1)
    
    V1 = U[k[0]:n+1+k[0]]
    V2 = U[k[1]:n+1+k[1]]
    V3 = U[k[2]:n+1+k[2]]
    V4 = U[k[3]:n+1+k[3]]
    
    fn1 = proj_B_Kr(1,n,V1,V2,V3,V4,B,C,T, params, omegas)
    fn2 = proj_B_Kr(2,n,V1,V2,V3,V4,B,C,T, params, omegas)
    fn3 = proj_B_Kr(3,n,V1,V2,V3,V4,B,C,T, params, omegas)
    fn4 = proj_B_Kr(4,n,V1,V2,V3,V4,B,C,T, params, omegas)

    #Fn= inv(MN)*[fn1;fn2;fn3;fn4];

    return np.concatenate([fn1,fn2,fn3,fn4])

        
def basis(m, n):
    
    Phi = np.zeros([np.size(m),n+1])
    
    for ii in range(np.size(m)): 
        for jj in range(n+1):
            
            if (jj == 0 and m[ii] < 1./n):
                Phi[ii ,jj] = (n*m[ii] - jj + 1)
              
            if ( m[ii] > (jj - 1.)/n and m[ii] < (jj + 1.)/n):
                Phi[ii ,jj] = (m[ii] <= 1.*jj/n)*(n*m[ii] - jj + 1) + (m[ii] > 1.*jj/n)*(-1*n*m[ii] + jj + 1)
                
            if (jj == n and m[ii] > 1-(1./n)):
                Phi[ii ,jj] = (n*m[ii] - jj + 1)
    return Phi
