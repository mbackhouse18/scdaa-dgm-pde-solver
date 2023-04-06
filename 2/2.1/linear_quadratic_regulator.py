import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import torch

class LQRSolver:

    def __init__(self, H, M, D, C, R, sigma, T):
        self.H = H.double()
        self.M = M.double()
        self.D = D.double()
        self.C = C.double()
        self.R = R.double()
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
        
        # Rewrite the matrix ode as a 1*4 vector
        dQ_dt = dQ_dt_matrix.reshape(4,)
        
        return dQ_dt

    
    def riccati_solver(self, time_grid):
        '''
        Solve the Riccati odes of Q(t), then do the time-reversal to get S(t), return time and 
        associated S(t) values (tuple, 1*4)
        '''
        if type(time_grid) == torch.Tensor:
            time_grid = time_grid.numpy()

        Q_0 = self.R.reshape(4,) # initial condition: Q(0)=S(T)=R

        # Solving S(r) on [t,T] is equivalent to solving Q(r)=S(T-r) on [0,T-t] 
        time_grid_Q = np.flip(self.T-time_grid) 
        interval = np.array([time_grid_Q[0], time_grid_Q[-1]]) 
        sol = integrate.solve_ivp(self.riccati_ode, interval, Q_0, t_eval=time_grid_Q)

        t_val = self.T - sol.t # do the time-reversal to get the solution S(t)

        return np.flip(t_val), np.flip(sol.y)
        
    def riccati_plot(self, time_grid):
        '''
        Plot the solutions S(t)
        '''
        sol_t, sol_y = self.riccati_solver(time_grid)
        plt.plot(sol_t,sol_y[0],label='S[0,0]',color='blue')
        plt.plot(sol_t,sol_y[1],label='S[0,1]',color='red')
        plt.plot(sol_t,sol_y[2],label='S[1,0]',color='yellow')
        plt.plot(sol_t,sol_y[3],label='S[1,1]',color='purple')

        plt.xlabel('time')
        plt.ylabel('S(t)')
        plt.legend(['S[0,0]','S[0,1]','S[1,0]','S[1,1]'])
        plt.show()

    def value_function(self, t, x):
        '''
        Input:
        1) t: time at which you want to compute the value function; torch tensor, dimension = batch size;
        2) x: spatial variable, each component of x is a 1*2 vector; torch tensor, dimension = batch size*1*2;
           Remark, here our component of x is actually the transpose of the original
           x in the problem, x_here: 1*2, x_original: 2*1
        Output:
        v(t,x): values of the value function evaluated at (t,x)
        '''
        n = 500 # Fix the number of steps to be 500
        val_func = torch.zeros((len(x),1), dtype=torch.float64) 
        x = x.double()

        for j in range(len(x)):
            initial_time = t[j].double().item() 
            step = (self.T-initial_time)/n # step = (T-t)/n
            time_grid = torch.arange(initial_time, self.T+step, step) # generate the time grid on [t,T]
            t_val, S_r = self.riccati_solver(time_grid)   
            S_t = torch.tensor([[S_r[0,0], S_r[1,0]], [S_r[2,0], S_r[3,0]]]) 
            S_t = S_t.double()

            # Assuming sigma is 2x1
            sig = torch.matmul(self.sigma, self.sigma.t()) # 2*2 matrix
            sig = sig.double()
            integral = 0
            for i in range(len(t_val)-1):
                S_i = torch.tensor([[S_r[0,i], S_r[1,i]], [S_r[2,i], S_r[3,i]]])
                S_i_1 = torch.tensor([[S_r[0,i+1], S_r[1,i+1]], [S_r[2,i+1], S_r[3,i+1]]])
                difference = S_i_1-S_i
                integral += torch.trace(torch.matmul(sig,difference))*(t_val[i+1] - t_val[i])

            x_j = x[j].reshape(1,-1).t()
            x_j_t = x_j.t()
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
        n = 500
        a_star = torch.zeros(len(x), 2)
        x = x.double()

        for i in range(len(x)):
            init_time = t[i].double().item() 
            step = (self.T-init_time)/n # step = (T-t)/n
            time_grid = torch.arange(init_time, self.T+step, step) # generate the time grid on [t,T]
            S_r = self.riccati_solver(time_grid)[1]
            S_t = torch.tensor([[S_r[0,0], S_r[1,0]], [S_r[2,0], S_r[3,0]]]) 
            S_t = S_t.double()
            x_i = x[i].reshape(1,-1).t() 

            # The product is 2*1, need to flatten it first before appending the value to a_star
            a_star[i] = -torch.flatten(torch.linalg.multi_dot([self.D,self.M.t(),S_t,x_i])) 
            
        return a_star
