import numpy as np
import torch
import matplotlib.pyplot as plt
from linear_quadratic_regulator import LQRSolver
from value_simulator import LQR_simulator
csfont = {'fontname':'Times New Roman'}
plt.rcParams['text.usetex'] = True

########################################################################################
############################ Initializing the calss ####################################
########################################################################################
T=1.0
H = torch.tensor([[1.0,0],[0,1.0]]) # H,M,D,C,R:  dim = 2*2 
M = torch.tensor([[1.0,0],[0,1.0]])
D = torch.tensor([[0.1,0],[0,0.1]])
C = torch.tensor([[0.1,0],[0,0.1]])
R = torch.tensor([[1.0,0],[0,1.0]])
sigma = torch.tensor([0.05, 0.05]) # sigma: dim = 2

val_simulator = LQR_simulator(T, H, M, C, D, R, sigma)
lqr_solver = LQRSolver(H, M, D, C, R, sigma, T)

########################################################################################################
######################################## Sampling ######################################################
########################################################################################################
batch_size = 100 # We will test v(t,x) at 100 points, this will take around 1.5 hours to run for fixed 
# number of Monte Carlo samples case, and approximately 3.5 hours for fixed time steps case. 
t = torch.rand(batch_size) # 0=<t<1, dim = batchsize
x1 = -4 * torch.rand([batch_size, 1]) + 2 # x1  uniformaly distributed on [-2,2) dim=n_points*1
x2 = -4 * torch.rand([batch_size, 1]) + 2 # x2  uniformaly distributed on [-2,2)
x = torch.cat((x1, x2), dim=1).unsqueeze(1) # dim = n_points*1*2 
val_theo = lqr_solver.value_function(t, x)

# Plot the sample points
# plt.scatter(x[:,0,0],x[:,0,1],color='dodgerblue',alpha=0.8,marker='o',edgecolor='black')
# plt.title(r'$\left(x_{1}^{i}, x_{2}^{i}\right)$', loc='center')
# plt.title(f'$i$={batch_size}', loc='right')
# plt.xlabel(r'$x_{1}$')
# plt.ylabel(r'$x_{2}$')
# plt.show()

#########################################################################################################
################################## Fixed Monte Carlo Samples ############################################
#################################### Different Time Steps ###############################################
#########################################################################################################
#########################################################################################################
Q = 10**5 # number of Monte Carlo samples
N = [1, 10, 50, 100, 500, 1000, 5000] # number of time steps
error_N = []
for n in N:
    val_pred = val_simulator.val_sim(t, x, n, Q)
    error_N.append(torch.max(torch.abs((val_theo-val_pred))).item())

# Plot the error in the log-log frame
plt.figure(figsize=(8,6))
plt.loglog(N, error_N, '-o', color = 'royalblue')
plt.title(f'Monte Carlo Samples = 100000, Batchsize = {batch_size}', **csfont)
plt.xlabel('number of time steps (log)', **csfont)
plt.ylabel(r'$L^{\infty}$-error (log)', **csfont)
plt.show()


#########################################################################################################
######################################### Fixed Time Steps ##############################################
################################# Different Monte Carlo Samples #########################################
#########################################################################################################
#########################################################################################################
# Q_prime = [10, 50, 10**2, 5*10**2, 10**3, 5*10**3, 10**4, 5*10**4, 10**5]
# N_prime = 5000
# error_Q = []
# for q in Q_prime:
#     value_pred = val_simulator.val_sim(t, x, N_prime, q)
#     error_Q.append(torch.max(torch.abs((val_theo-value_pred))).item())

# Plot the error in the log-log frame
# plt.figure(figsize=(8,6))
# plt.loglog(Q_prime, error_Q, '-o', color = 'purple')
# plt.title(f'Number of Time Steps = 5000, Batchsize = {batch_size}', **csfont)
# plt.xlabel('number of Monte Carlo samples (log)', **csfont)
# plt.ylabel(r'$L^{\infty}$-error (log)', **csfont)
# plt.show()



