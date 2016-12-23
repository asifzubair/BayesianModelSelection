import numpy as np
from scipy import linalg
from models_utils import *


class PapaModel:
    def __init__(self):
    	## self.name = name
        self.n = 99
        self.k = 20
        self.tau = 2.25
        self.t = self.k*self.tau
        self.N = 100
        self.ab = 7.3

        Hbdomain =  np.zeros(100)
        Hbdomain[30:70] =  1
        Knidomain = np.zeros(100)
        Knidomain[40:90] = 1
        Krdomain = np.zeros(100)
        Krdomain[20:80] = 1
        Gtdomain = np.zeros(100)
        Gtdomain[10:90] = 1
        self.domains = np.concatenate([Hbdomain,Knidomain,Krdomain,Gtdomain])

        Basis0 = basis(np.linspace(0,1,self.N), self.n)
        self.Basis = linalg.block_diag(Basis0,Basis0,Basis0,Basis0)

        inFile = np.loadtxt('inputs-outputs.txt', delimiter='\t')
        f10 = inFile[:,1]
        f20 = np.zeros(self.N)
        f30 = np.zeros(self.N)
        f40 = np.zeros(self.N)
        self.f0 = np.concatenate([f10,f20,f30,f40]) 
        self.y = np.concatenate([inFile[:,3],inFile[:,6],inFile[:,5],inFile[:,4]])
        self.B = inFile[:,0]
        self.T = inFile[:,2]

        Mn = tridiag(1./(6*self.n)*np.ones(self.n),2./(3*self.n)*np.ones(self.n+1),1./(6*self.n)*np.ones(self.n))
        Mn[0,0] = 1./(3*self.n) 
        Mn[self.n,self.n] = 1./(3*self.n)
        Kn = tridiag(-self.n*np.ones(self.n),2*self.n*np.ones(self.n+1),-self.n*np.ones(self.n))
        Kn[0,0] = self.n
        Kn[self.n,self.n] = self.n
        
        self.In = np.eye(self.n+1)
        self.An = linalg.inv(Mn).dot(Kn)
        self.U0 = self.f0

    def predict(self, parms):
        alpha, D, Co, Ns, K, K1, K2, K3 = parms
        K=10**(K - self.ab)
        K1=10**(K1 - self.ab)
        K2=10**(K2 - self.ab)
        K3=10**(K3 - self.ab)
        Co=10**(Co)
        # D = (D)/250000
        D = (D*45.0)/250000
        ## alpha = 45.*alpha
        beta = alpha
        params = [alpha, K, K1, K2, K3, Co, Ns]
        C = (1.*Co)/(Co + (1+Co*K*self.B)**Ns - 1)

        omega1=1./(max(omega(1, self.f0[0:self.N], self.B, C, self.T, params)))
        omega2=1./(max(omega(2, self.f0[0:self.N], self.B, C, self.T, params)))
        omega3=1./(max(omega(3, self.f0[0:self.N], self.B, C, self.T, params)))
        omega4=1./(max(omega(4, self.f0[0:self.N], self.B, C, self.T, params)))
        omegas = [omega1, omega2, omega3, omega4]
    
        U = np.zeros([4*(self.n+1),self.k])
        U[:,0] = self.U0
    
        phi = linalg.expm((-1*D*self.An-beta*self.In)*self.tau)
        An_inv = linalg.inv(-1*D*self.An-beta*self.In)
        AN_INV = linalg.block_diag(An_inv,An_inv,An_inv,An_inv)
        Phi =  linalg.block_diag(phi,phi,phi,phi)    
        BN = AN_INV.dot(Phi-np.eye(4*(self.n+1)))
    
        for ii in range(1,self.k):
            Fn = nonlinear(self.n, U[:,ii-1], self.B, C, self.T, params, omegas)
            U[:,ii] = Phi.dot(U[:,ii-1]) + BN.dot(Fn)
        return U[:,self.k-1]

class PapaModelA4(PapaModel):
    def __init__(self):
        self.name = "A4"
        PapaModel.__init__(self)
    def predict(self, parms):
        alpha, Co, K, K1 = parms
        D = 0; Ns = 3; K2 = K3 = K;
        params = [alpha, D, Co, Ns, K, K1, K2, K3]
        return PapaModel.predict(self, params)

class PapaModelA6(PapaModel):
    def __init__(self):
        self.name = "A6"
        PapaModel.__init__(self)
    def predict(self, parms):
        return PapaModel.predict(self, np.append(parms, [parms[5], parms[4]]))

class PapaModelB7(PapaModel):
    def __init__(self):
        self.name = "B7"
        PapaModel.__init__(self)
    def predict(self, parms):
        return PapaModel.predict(self, np.append(parms, parms[4]))

class PapaModelB7r(PapaModel):
    def __init__(self):
        self.name = "B7r"
        PapaModel.__init__(self)
    def predict(self, parms):
       alpha, D, Co, Ns, K, K1, K3 = parms
       K2 = K1;
       params = np.array([alpha, D, Co, Ns, K, K1, K2, K3])
       return PapaModel.predict(self, params)

class PapaModelC8(PapaModel):
    def __init__(self):
        self.name = "C8"
        PapaModel.__init__(self)
    def predict(self, parms):
        return PapaModel.predict(self, parms)

class PapaModel_B_Kr(PapaModel):
    def __init__(self):
        self.name = "B_Kr"
        PapaModel.__init__(self)
    def predict(self, parms):
        alpha, D, Co, Ns, K, K1, K2, K3 = parms
        K=10**(K - self.ab)
        K1=10**(K1 - self.ab)
        K2=10**(K2 - self.ab)
        K3=10**(K3 - self.ab)
        Co=10**(Co)
        D = (D*45.0)/250000
        beta = alpha
        params = [alpha, K, K1, K2, K3, Co, Ns]
        C = (1.*Co)/(Co + (1+Co*K*self.B)**Ns - 1)
        
        omega1=1./(max(omega_B_Kr(1, self.f0[0:self.N], self.B, C, self.T, params)))
        omega2=1./(max(omega_B_Kr(2, self.f0[0:self.N], self.B, C, self.T, params)))
        omega3=1./(max(omega_B_Kr(3, self.f0[0:self.N], self.B, C, self.T, params)))
        omega4=1./(max(omega_B_Kr(4, self.f0[0:self.N], self.B, C, self.T, params)))
        omegas = [omega1, omega2, omega3, omega4]
    
        U = np.zeros([4*(self.n+1),self.k])
        U[:,0] = self.U0
        
        phi = linalg.expm((-1*D*self.An-beta*self.In)*self.tau)
        An_inv = linalg.inv(-1*D*self.An-beta*self.In)
        AN_INV = linalg.block_diag(An_inv,An_inv,An_inv,An_inv)
        Phi =  linalg.block_diag(phi,phi,phi,phi)    
        BN = AN_INV.dot(Phi-np.eye(4*(self.n+1)))
    
        for ii in range(1,self.k):
            Fn = nonlinear_B_Kr(self.n, U[:,ii-1], self.B, C, self.T, params, omegas)
            U[:,ii] = Phi.dot(U[:,ii-1]) + BN.dot(Fn)
        return U[:,self.k-1]
        
class PapaModel_B_Kr7(PapaModel_B_Kr):
    def __init__(self):
        self.name = "B_Kr7"
        PapaModel.__init__(self)    
    def predict(self, parms):
        return PapaModel_B_Kr.predict(self, np.append(params, params[4]))

class PapaModel_B_Kr8(PapaModel_B_Kr):
    def __init__(self):
        self.name = "B_Kr8"
        PapaModel.__init__(self)
    def predict(self, parms):
        return PapaModel_B_Kr.predict(self, params)
