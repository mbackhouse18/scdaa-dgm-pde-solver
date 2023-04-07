from linear_quadratic_regulator import LQRSolver
from value_sim import value_function_sim
import numpy as np
import torch
import matplotlib.pyplot as plt
csfont = {'fontname':'Times New Roman'}

H = torch.tensor([[1.0,0],[0,1.0]]) # H,M,D,C,R:  dim = 2*2 
M = torch.tensor([[1.0,0],[0,1.0]])
D = torch.tensor([[0.1,0],[0,0.1]])
C = torch.tensor([[0.1,0],[0,0.1]])
R = torch.tensor([[1.0,0],[0,1.0]])
sigma = torch.tensor([0.05, 0.05]) # sigma: dim = 2


# Setting the time horizon
T = 1.0 # time horizon [0,T]
# We test the value function at 10 randomly generated points
n_points = 10
t = torch.rand(n_points) # t uniformaly distributed on [0,1)
x1 = -4 * torch.rand([n_points, 1]) + 2 # x1  uniformaly distributed on [-2,2)
x2 = -4 * torch.rand([n_points, 1]) + 2 # x2  uniformaly distributed on [-2,2)
x = torch.cat((x1, x2), dim=1).unsqueeze(1) # dim = n_points*1*2

# Initialize the simulators, our simulator only works for 1 point each time
value_simulator_1 = value_function_sim(t[0].item(), x[0].squeeze(0), T, H, M, C, D, R, sigma)
value_simulator_2 = value_function_sim(t[1].item(), x[1].squeeze(0), T, H, M, C, D, R, sigma)
value_simulator_3 = value_function_sim(t[2].item(), x[2].squeeze(0), T, H, M, C, D, R, sigma)
value_simulator_4 = value_function_sim(t[3].item(), x[3].squeeze(0), T, H, M, C, D, R, sigma)
value_simulator_5 = value_function_sim(t[4].item(), x[4].squeeze(0), T, H, M, C, D, R, sigma)
value_simulator_6 = value_function_sim(t[5].item(), x[5].squeeze(0), T, H, M, C, D, R, sigma)
value_simulator_7 = value_function_sim(t[6].item(), x[6].squeeze(0), T, H, M, C, D, R, sigma)
value_simulator_8 = value_function_sim(t[7].item(), x[7].squeeze(0), T, H, M, C, D, R, sigma)
value_simulator_9 = value_function_sim(t[8].item(), x[8].squeeze(0), T, H, M, C, D, R, sigma)
value_simulator_10 = value_function_sim(t[9].item(), x[9].squeeze(0), T, H, M, C, D, R, sigma)
lqr_equation = LQRSolver(H, M, D, C, R, sigma, T)
val_exact = lqr_equation.value_function(t, x)  # dim=n_point*1

########################################################################################################
################################# Fixed Monte Carlo Samples ############################################
################################### Different Time Steps ###############################################
########################################################################################################
########################################################################################################


Q = 10**5 # number of Monte Carlo samples
N = [1, 10, 50, 100, 500, 1000, 5000] # number of time steps
error_N = [] 
for n in N:
    err_n_1 = np.abs(value_simulator_1.value_simulation(n, Q).item()-val_exact[0].item())
    err_n_2 = np.abs(value_simulator_2.value_simulation(n, Q).item()-val_exact[1].item())
    err_n_3 = np.abs(value_simulator_3.value_simulation(n, Q).item()-val_exact[2].item())
    err_n_4 = np.abs(value_simulator_4.value_simulation(n, Q).item()-val_exact[3].item())
    err_n_5 = np.abs(value_simulator_5.value_simulation(n, Q).item()-val_exact[4].item())
    err_n_6 = np.abs(value_simulator_6.value_simulation(n, Q).item()-val_exact[5].item())
    err_n_7 = np.abs(value_simulator_7.value_simulation(n, Q).item()-val_exact[6].item())
    err_n_8 = np.abs(value_simulator_8.value_simulation(n, Q).item()-val_exact[7].item())
    err_n_9 = np.abs(value_simulator_9.value_simulation(n, Q).item()-val_exact[8].item())
    err_n_10 = np.abs(value_simulator_10.value_simulation(n, Q).item()-val_exact[9].item())
    error_N.append(max(err_n_1,err_n_2,err_n_3,err_n_4,err_n_5,err_n_6,err_n_7,err_n_8,err_n_9,err_n_10))

# Plot the error in the log-log frame, takes 20 mins to run for simulating 10 points
plt.figure(figsize=(8,6))
plt.loglog(N, error_N, '-o', color = 'royalblue')
plt.title('Monte Carlo Samples = 100000, Batchsize = 10', **csfont)
plt.xlabel('number of time steps (log)', **csfont)
plt.ylabel('L-infinity error (log)', **csfont)
plt.show()


########################################################################################################
######################################## Fixed Time Steps ##############################################
################################ Different Monte Carlo Samples #########################################
########################################################################################################
########################################################################################################


Q_prime = [10, 50, 10**2, 5*10**2, 10**3, 5*10**3, 10**4, 5*10**4, 10**5]
N_prime = 5000
error_Q = []
for q in Q_prime:
    err_q_1 = np.abs(value_simulator_1.value_simulation(N_prime, q).item()-val_exact[0].item())
    err_q_2 = np.abs(value_simulator_2.value_simulation(N_prime, q).item()-val_exact[1].item())
    err_q_3 = np.abs(value_simulator_3.value_simulation(N_prime, q).item()-val_exact[2].item())
    err_q_4 = np.abs(value_simulator_4.value_simulation(N_prime, q).item()-val_exact[3].item())
    err_q_5 = np.abs(value_simulator_5.value_simulation(N_prime, q).item()-val_exact[4].item())
    err_q_6 = np.abs(value_simulator_6.value_simulation(N_prime, q).item()-val_exact[5].item())
    err_q_7 = np.abs(value_simulator_7.value_simulation(N_prime, q).item()-val_exact[6].item())
    err_q_8 = np.abs(value_simulator_8.value_simulation(N_prime, q).item()-val_exact[7].item())
    err_q_9 = np.abs(value_simulator_9.value_simulation(N_prime, q).item()-val_exact[8].item())
    err_q_10 = np.abs(value_simulator_10.value_simulation(N_prime, q).item()-val_exact[9].item())
    error_Q.append(max(err_q_1, err_q_2, err_q_3, err_q_4, err_q_5, err_q_6, err_q_7, err_q_8, err_q_9, err_q_10))

# Plot the error in the log-log frame, 30 mins to run for simulating 10 points
plt.figure(figsize=(8,6))
plt.loglog(Q_prime, error_Q, '-o', color = 'crimson')
plt.title('Number of Time Steps = 5000, Batchsize = 10', **csfont)
plt.xlabel('number of Monte Carlo samples (log)', **csfont)
plt.ylabel('L-infinity error (log)', **csfont)
plt.show()




