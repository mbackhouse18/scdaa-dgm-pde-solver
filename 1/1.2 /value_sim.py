import numpy as np
import torch
import matplotlib.pyplot as plt
from linear_quadratic_regulator import LQRSolver

class value_function_sim:
    '''
    Simulate the value function at one given (t,x) on [0,T]*R*R 
    '''
    def __init__(self, t, x, T, H, M, C, D, R, sigma):
        self.H, self.M, self.C = H.double(), M.double(), C.double()
        self.D, self.R, self.sigma = D.double(), R.double(), sigma.double()
        self.t, self.x, self.T = t, x, T

    def mat_ext(self, mat, size):
        '''
        Extend the matrices from dim = 2*2 to dim = number of Monte Carlo samples (Q) *2*2
        '''
        if mat.shape == torch.Size([2, 2]):
            return mat.unsqueeze(0).repeat(size,1,1)
        elif mat.shape == torch.Size([1, 2]):
            return mat.t().unsqueeze(0).repeat(size,1,1)
        elif mat.shape == torch.Size([2]):
            return mat.t().unsqueeze(0).unsqueeze(1).repeat(size,1,1) 
        
    def value_simulation(self, n_step, n_Monte_Carlo):
        '''
        n_step, n_Monte_Carlo: integer
        t, T: float 
        x: torch tensor, dim = 1*2
        H, M, C, D, R: torch.tensor, dim = 2*2
        sigma = torch.tensor, dim = 2
        '''
        H, M, C, D, R, sigma = self.H, self.M, self.C, self.D, self.R, self.sigma
        t, x, T = self.t, self.x, self.T
        k = int(t/T*n_step)-1
        tau = T/n_step
        time_grid = torch.linspace(t,T,n_step-k)
        lqr_equation = LQRSolver(H, M, D, C, R, sigma, T)
        S_r = lqr_equation.riccati_solver(time_grid)[1]
        S_r = torch.from_numpy(S_r.copy()).double()

        MDM = self.mat_ext(torch.linalg.multi_dot([M,torch.inverse(D),M.t()]),n_Monte_Carlo)
        DM = self.mat_ext(torch.matmul(torch.inverse(D),M.t()),n_Monte_Carlo)
        sig = self.mat_ext(torch.sqrt(torch.tensor(tau))*sigma.t(),n_Monte_Carlo)
    
    
        X = torch.zeros((len(time_grid),n_Monte_Carlo,1,2),dtype=torch.float64) # dim=(N-k)*Q*1*2
        X[0,:] = x # initial value X_t = x
        integral = torch.zeros(n_Monte_Carlo,1,dtype=torch.float64) # dim=Q*1
        for i in range(1,len(time_grid)):
            # HX & MDMSX terms
            HX = torch.bmm(self.mat_ext(H,n_Monte_Carlo), X[i-1].reshape(n_Monte_Carlo,2,1)).reshape(n_Monte_Carlo,1,2) # dim = Q*1*2
            MDMSX = torch.bmm(MDM,torch.bmm(self.mat_ext(S_r[i-1],n_Monte_Carlo),X[i-1].reshape(n_Monte_Carlo,2,1))).reshape(n_Monte_Carlo,1,2) # dim = Q*1*2
        
            # Stochastic term
            delta_W = torch.randn((n_Monte_Carlo,1,1),dtype=torch.float64) # dim = Q*1*1
            stoch = torch.bmm(sig.reshape(n_Monte_Carlo,2,1),delta_W) # dim = Q*2*1
            stoch = stoch.reshape(n_Monte_Carlo,1,2) # dim = Q*1*2
        
            # X_i
            X[i] = X[i-1]+tau*(HX-MDMSX)+stoch #dim = Q*1*2

            # Computing the objective function J
            term_1 = (torch.bmm(X[i],torch.bmm(self.mat_ext(C,n_Monte_Carlo),X[i].reshape(n_Monte_Carlo,2,1)))\
                      +torch.bmm(X[i-1],torch.bmm(self.mat_ext(C,n_Monte_Carlo),X[i-1].reshape(n_Monte_Carlo,2,1))))/2 # dim = Q*1*1
            term_1 = term_1.squeeze(2) # dim = Q*1

            a_i = -torch.bmm(DM,torch.bmm(self.mat_ext(S_r[i],n_Monte_Carlo),X[i].reshape(n_Monte_Carlo,2,1))) # dim = Q*2*1
            a_i_1 = -torch.bmm(DM,torch.bmm(self.mat_ext(S_r[i-1],n_Monte_Carlo),X[i-1].reshape(n_Monte_Carlo,2,1)))
            term_2 = (torch.bmm(a_i.reshape(n_Monte_Carlo,1,2),torch.bmm(self.mat_ext(D,n_Monte_Carlo),a_i))\
                     +torch.bmm(a_i_1.reshape(n_Monte_Carlo,1,2),torch.bmm(self.mat_ext(D,n_Monte_Carlo),a_i_1)))/2 # dim = Q*1*1
            term_2 = term_2.squeeze(2) # dim = Q*1

            integral += (term_1+term_2)*tau
        integral += torch.bmm(X[-1],torch.bmm(self.mat_ext(R,n_Monte_Carlo),X[-1].reshape(n_Monte_Carlo,2,1))).squeeze(1) 
            
        value_simulation = torch.mean(integral) # Taking the conditional expectation E[J|X_t=x]

        return value_simulation
