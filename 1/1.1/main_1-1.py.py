from linear_quadratic_regulator import LQRSolver
import numpy as np
import torch
import random
import matplotlib.pyplot as plt

# random.seed(42)


H = torch.tensor([[1.0,0],[0,1.0]]).double()
M = torch.tensor([[1.0,0],[0,1.0]]).double()
D = torch.tensor([[0.1,0],[0,0.1]]).double()
C = torch.tensor([[0.1,0],[0,0.1]]).double()
R = torch.tensor([[1.0,0],[0,1.0]]).double()
sigma = torch.tensor([0.05, 0.05]) # 1*2, will be automatically reshaped as 2*1 by the class

T = 1.0
t0 = torch.tensor([0.5,0.5]).double() # batchsize = 3
# n=500, number of steps, defined inside the class

x = torch.tensor([[[0.5, 0.8]],
                  [[3, 2]],]).double()

#######################################################################################
###################################### Test############################################
#######################################################################################
# Initializing 
lqr_equation = LQRSolver(H, M, D, C, R, sigma, T)

#######################################################################################
# Testing the method riccati_solver 
# Input: time grid, when calling the method riccati_solver and riccati_plot, we need 
# to specify the time grid
# Output: time points 1*n; S(t): 4*n 

# Generate a time gird for testing
time_grid_test = np.linspace(0.0,1.0,1000)
sol_t, sol_S = lqr_equation.riccati_solver(time_grid_test)
#print(sol_t)
#print(sol_S)
#print(len(sol_t))
#print(len(sol_S))

#######################################################################################
# Testing the method riccati_plot
# Output: plot of the solutions

lqr_equation.riccati_plot(time_grid_test)


#######################################################################################
# Testing the method value_function
# Input: time: batchsize; x: batchsize*1*2
# Output: value function: batchsize*1
# n=500 is predefined in this method, no need to generate the time grid

value_funs = lqr_equation.value_function(t0,x)
print(f"Value Function: {value_funs}")

#######################################################################################
# Testing the method optimal_control
# Input: time: batchsize; x: batchsize*1*2
# Output: optimal control: batchsize*2
# n=500 is predefined in this method, no need to generate the time grid

optimal_control = lqr_equation.optimal_control(t0,x)
print(f"Optimal Control: {optimal_control}")



