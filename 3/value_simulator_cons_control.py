import numpy as np
import torch
import matplotlib.pyplot as plt
from linear_quadratic_regulator import LQRSolver

class LQR_simulator_cons_control:
    def __init__(self, T, H, M, C, D, R, sigma):
        '''
        Initialization
        '''
        self.H, self.M, self.C = H.double(), M.double(), C.double() # dim = 2*2
        self.D, self.R = D.double(), R.double()
        self.T = T # float number
    
        if sigma.shape == torch.Size([1, 2]):
            self.sigma = sigma.t().double()
        elif sigma.shape == torch.Size([2]):
            self.sigma = sigma[:,None].double()
        elif sigma.shape == torch.Size([2,1]):
            self.sigma = sigma.double()
        else:
            raise ValueError('Sigma must be torch tensor with dimension 1*2, 2*1 or 2!')
        
      
    def control(self, X):
        '''
        Return the constant control [1,1], dim = len(X)*2*1
        '''
        return torch.tensor([[1],[1]])[None,:].repeat(len(X),1,1).double()
    
        
    def value_func_sim(self, t, x, n_step, n_Monte_Carlo):
        '''
        Numerically solve the controlled SDE: dX_s = [HX_s+Ma]ds + sigma*dW, t<=s<=T with initial condition X_t = x,
        where:
             1) t: float 
             2) x: torch tensor with dimension 1*2
             3) a = -D^{-1}M^{T}S(t)x
        '''
        if x.shape == torch.Size([2]):
            x = x[:,None]
        elif x.shape == torch.Size([1, 2]):
            x = x.t() # dim = 
        elif x.shape == torch.Size([1, 2]):
            x = x
        else:
            raise ValueError('x must be torch tensor with dimension 2*1, 1*2 or 2!')
        
        value_tx = 0 

        if t == self.T:
            value_tx = (x.t()@self.R@x).item()
        else:

            k = int(t/self.T*n_step)
            tau = self.T/n_step
            time_grid = torch.linspace(0,self.T,n_step+1) # time_grid = [t,T] with step tau
            lqr_equation = LQRSolver(self.H, self.M, self.D, self.C, self.R, self.sigma, self.T)
        
            time_grid_init = time_grid[k:]
            # Solve the associated Riccati equation with terminal condition S[T]=R, dim = n_step*2*2
            S_r = lqr_equation.riccati_solver(time_grid_init)[1] 
            S_r = torch.from_numpy(S_r.copy()).double() 
        
            X = torch.zeros(len(time_grid_init), n_Monte_Carlo, 2, 1, dtype = torch.float64) # dim = (n_step-k)*n_Monte_Carlo*2*1
            #x = x.t() # dim = 2*1
            X[0] = x[None,:,:].repeat(n_Monte_Carlo, 1,1) # dim = n_Monte_Carlo*2*1, initial condition X_t = x
        
            stoch_coef = self.sigma*np.sqrt(tau) # dim = 2*1 
        
            integral = torch.zeros(n_Monte_Carlo,1,1) # dim = n_Monte_Carlo*1*1    
            for i in range(1,len(time_grid_init)):
                # X[i]: dim = n_Monte_Carlo*2*1 
            
                a_i_1 = self.control(X[i-1])
                X[i] = X[i-1] + tau*(self.H[None,:]@X[i-1]\
                        +(self.M@a_i_1))\
                        +stoch_coef@torch.randn(n_Monte_Carlo,1,1, dtype = torch.float64)
                a_i = self.control(X[i]) # dim = n_Monte_Carlo*2*1 
                # J
                int_1st = (X[i].reshape(n_Monte_Carlo,1,2)@self.C@X[i]+X[i-1].reshape(n_Monte_Carlo,1,2)@self.C@X[i-1])/2
                int_2nd = (a_i.reshape(n_Monte_Carlo,1,2)@self.D@a_i+a_i_1.reshape(n_Monte_Carlo,1,2)@self.D@a_i_1)/2
                integral += (int_1st+int_2nd)*tau
            
            integral += X[-1].reshape(n_Monte_Carlo,1,2)@self.R@X[-1]
            value_tx = torch.mean(integral).item()
            
        return value_tx
        
    def val_sim(self, t, x, n_step, n_Monte_Carlo):
        '''
         Simulate the value function at given points, where
         1) t: dim = number of points
         2) x: dim = number of points*1*2
        '''
        val_sim = torch.zeros(len(t),1)
        for ind, (_t, _x) in enumerate(zip(t, x)):
            t = _t.item()
            val_sim[ind] = torch.tensor(self.value_func_sim(_t.item(), _x, n_step, n_Monte_Carlo))
        return val_sim   
        
        
