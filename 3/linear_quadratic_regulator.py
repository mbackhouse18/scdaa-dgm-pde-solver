import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import torch
csfont = {'fontname':'Times New Roman'}
plt.rcParams['text.usetex'] = True


class LQRSolver:

    def __init__(self, H, M, D, C, R, sigma, T):
        self.H, self.M, self.D, self.C, self.R = H.double(), M.double(), D.double(), C.double(), R.double()
        self.sigma = sigma.reshape(1,-1).t() # reshape the input sigma as a 2*1 matrix
        self.T = T

    def riccati_ode(self, t, Q):
        '''
        Definition of the Riccati ode satisfied by Q(t):=S(T-t), Q(0)=S(T)=R, where Q, Q(0) are 2*2 functional matrices;
        since solve_ivp only support flattern variables, here our imput Q is a 1*4 array, i.e., an vector;
        Riccati ode: dQ(t)/dt = 2H^{T}Q(t)-Q(t)MD^{-1}MQ(t)+C 
        '''
        if type(self.C) == torch.Tensor:
             self.C = self.C.numpy()

        # Rewrite the imput 1*4 vector as a 2*2 matrix
    
        Q_matrix = Q.reshape((2,2))
        
        # RHS of the ode
        quadratic_term = -np.linalg.multi_dot([Q_matrix,self.M,np.linalg.inv(self.D),self.M,Q_matrix])
        linear_term = 2*np.dot(np.transpose(self.H),Q_matrix)
        constant_term = self.C
        
        # Riccati ode in the matrix form
        dQ_dt_matrix = linear_term + quadratic_term + constant_term
        
        return dQ_dt_matrix.flatten()

    
    def riccati_solver(self, time_grid):
        '''
        Solve the Riccati odes of Q(t), then do the time-reversal to get S(t), return time and 
        associated S(t) values (tuple, 1*4)
        '''
        if type(time_grid) == torch.Tensor:
            time_grid = time_grid.numpy()

        Q_0 = self.R.flatten() # initial condition: Q(0)=S(T)=R

        # Solving S(r) on [t,T] is equivalent to solving Q(r)=S(T-r) on [0,T-t] 
        time_grid_Q = np.flip(self.T-time_grid) 
        interval = np.array([time_grid_Q[0], time_grid_Q[-1]]) 
        sol = integrate.solve_ivp(self.riccati_ode, interval, Q_0, t_eval=time_grid_Q)

        t_val = self.T - sol.t # do the time-reversal to get the solution S(t)
        S_r = np.flip(sol.y,1).T.reshape(-1,2,2)
        return np.flip(t_val), S_r
        
    def riccati_plot(self, time_grid):
        '''
        Plot the solutions S(t)
        '''
        sol_t, sol_y = self.riccati_solver(time_grid)
        fig, axs = plt.subplots(2,2,sharex=True,sharey=True)
        axs[0,0].plot(sol_t,sol_y[:,0,0],label='S[1,1]',color='blue')
        axs[0,0].set(ylabel='S(t)', title=r'$S_{11}(t)$')
        axs[0,1].plot(sol_t,sol_y[:,0,1],label='S[1,2]',color='red')
        axs[0,1].set(title=r'$S_{12}(t)$')
        axs[1,0].plot(sol_t,sol_y[:,1,0],label='S[2,1]',color='yellow')
        axs[1,0].set(xlabel='time', ylabel='S(t)', title=r'$S_{21}(t)$')
        axs[1,1].plot(sol_t,sol_y[:,1,1],label='S[2,2]',color='orange')
        axs[1,1].set(xlabel='time', title=r'$S_{22}(t)$')
        fig.tight_layout()

        plt.show()

    def value_function(self, t, x):
        '''
        Input:
            1) t: time at which you want to compute the value function; torch tensor, dimension = batch size;
            2) x: space variable, each component of x is a 1*2 vector; torch tensor, dimension = batch size*1*2;
        Remark, here our component of x is actually the transpose of the original x in the problem, 
        x_here: 1*2, x_original: 2*1

        Output:
        v(t,x): values of the value function evaluated at (t,x)
        '''
        val_func = torch.zeros((len(x),1), dtype=torch.float64) 
        x = x.double()
        # Assuming sigma is 2x1
        sig = torch.matmul(self.sigma, self.sigma.t()) # sigma*sigma^T: 2*2 matrix
        sig = sig.double()
        
        for j in range(len(x)):
            init_t = t[j].double().item() 
            x_j = x[j].t()
            x_j_t = x[j]

            if init_t == self.T: 
                # At T, the value function is given by x^T*R*x
                val_func[j] = torch.linalg.multi_dot([x_j_t,self.R,x_j])
            else:
                time_grid = torch.linspace(init_t, self.T, 1000) # generate the time grid on [t,T]
                t_val, S_r = self.riccati_solver(time_grid)
                S_r = torch.from_numpy(S_r.copy()).double()  
                S_t = S_r[0] 

                integral = 0

                for i in range(1,len(t_val)):
                    diff = torch.trace(torch.matmul(sig,S_r[i])-torch.matmul(sig,S_r[i-1]))
                    integral += diff*(t_val[i] - t_val[i-1])
                val_func[j] = torch.linalg.multi_dot([x_j_t,S_t,x_j]) + integral

        return val_func
        

    def optimal_control(self, t, x):
        '''
        Input:
        1) t: time at which you want to compute the value function; torch tensor, dimension = batch size;
        2) x: spatial variable, each component of x is a 1*2 vector; torch tensor, dimension = batch size*1*2;
           Remark, here our component of x is actually the transpose of the original
           x in the problem, x_here: 1*2, x_original: 2*1
        Output:
        a(t,x): optimal control evaluated at (t,x), batchsize*2 for x two-dimensional
        '''
        a_star = torch.zeros(len(x), 2).double()
        x = x.double()

        for i in range(len(x)):
            init_t = t[i].double().item() 
            if init_t == self.T:
                a_star[i] = -torch.flatten(torch.linalg.multi_dot([torch.inverse(self.D),self.M.t(),self.R,x[i].t()])) 
            else:
                time_grid = torch.linspace(init_t, self.T, 1000) # generate the time grid on [t,T]
                S_r = self.riccati_solver(time_grid)[1] 
                S_t = torch.from_numpy(S_r[0].copy()).double() 
                # The product is 2*1, need to flatten it first before appending the value to a_star
                a_star[i] = -torch.flatten(torch.linalg.multi_dot([torch.inverse(self.D),self.M.t(),S_t,x[i].t()])) 
            
        return a_star
