from linear_quadratic_regulator import LQRSolver
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)

#######################################################################################
############################### Initializing the class ################################
#######################################################################################
# Coefficient matrices in the LQR
# H, M, D, C, R: dim = 2*2; sigma: dim = 1*2
H = torch.tensor([[1.0,0],[0,1.0]])
M = torch.tensor([[1.0,0],[0,1.0]])
D = torch.tensor([[0.1,0],[0,0.1]])
C = torch.tensor([[0.1,0],[0,0.1]])
R = torch.tensor([[1.0,0],[0,1.0]])
sigma = torch.tensor([0.05, 0.05]) 
# Time horizon [t,T]
T = 1.0

lqr_equation = LQRSolver(H, M, D, C, R, sigma, T)

#######################################################################################
################################# riccati_solver ######################################
#######################################################################################
# Input: time grid: ndarray
# Output: 1) time grid: ndarray  
#         2) S(t): ndarray, dim = len(time grids)*2*2 

time_grid_test = np.linspace(0.0,1.0,1000)
# sol_t, sol_S = lqr_equation.riccati_solver(time_grid_test)
# print(sol_t)
# print(sol_S)

# #######################################################################################
# ################################### riccati_plot ######################################
# #######################################################################################
# # Output: plot of the solutions

lqr_equation.riccati_plot(time_grid_test)

#######################################################################################
################################# value_function ######################################
#######################################################################################
# Input: 1) time: torch tensor, dim = batchsize
#        2) x: torch tensor, dim = batchsize*1*2
# Output: value function: torch tensor, dim = batchsize*1

t0 = torch.tensor([1.0,0.5,0.2]) # batchsize = 3
x = torch.tensor([[[0.5, 0.8]],
                  [[3, 2]],
                  [[1.5, -0.6]]])
value_funs = lqr_equation.value_function(t0,x)
print(f'v(t,x): {value_funs}')

#########################################################################################
################################# optimal_control #######################################
#########################################################################################
# Input: 1) time: torch tensor, dim = batchsize 
#        2) x: torch tensor, dim = batchsize*1*2
# Output: optimal control, torch tensor, dim = batchsize*2

t0 = torch.tensor([1.0,0.5,0.2]) # batchsize = 3
x = torch.tensor([[[0.5, 0.8]],
                  [[3, 2]],
                  [[1.5, -0.6]]])
optimal_control = lqr_equation.optimal_control(t0,x)
print(f'Optimal Control: {optimal_control}')



