import numpy as np
import matplotlib.pyplot as plt
import math
import random


from abc import ABC, abstractmethod

#DEFINE BASIS FUNCTIONS----------------------------------------------------
def BF_Gaussian(x, mu, gamma=1):
    return np.exp(-gamma * np.linalg.norm(mu-x)**2)


def BF_Polynomial(x,M):
    return pow(x,M)

#DEFINE LINEAR BASIS MODELS-----------------------------------------------

class LinearBasisFunctionModel(ABC):
   @abstractmethod
   def Compute_DesignMatrix(self):
      pass


class Polynomial(LinearBasisFunctionModel):
   def __init__(self, M,b = True):
      self.M = M
      self.b = b
   def Compute_DesignMatrix(self, X):
       # For Polynomial Basis
       Phi = np.ones((len(X), self.M))
       for m in range(self.M - self.b):
           Phi[:, m + self.b] = np.vectorize(BF_Polynomial)(X, m + self.b)
       return Phi

class Gaussian(LinearBasisFunctionModel):
   def __init__(self, M,samples_x, gamma = 0.1,b = True, auto_sigma = False, auto_mu = False):
       self.M = M
       self.b = b

       #Set Up Sigma auto sigma
       if auto_sigma:
           gamma = np.var(samples_x)
       self.gamma = gamma

        #Create center gaussians
       if auto_mu:
           idx = np.round(np.linspace(0, len(samples_x) - 1, M)).astype(int)
           self.mu = samples_x[idx]
       else:
           self.mu = np.zeros(M)

   def Compute_DesignMatrix(self, X):
       # For Gaussian Basis
       Phi = np.ones((len(X), self.M))
       for m in range(self.M - self.b):
           mu_ev = self.mu[m]
           Phi[:, m + self.b] = np.vectorize(BF_Gaussian)(X, mu_ev,self.gamma)
       return Phi
