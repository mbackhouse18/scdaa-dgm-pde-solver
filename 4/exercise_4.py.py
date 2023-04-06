import numpy as np
import torch
import torch.nn as nn
from linear_quadratic_regulator import LQRSolver
from functions import FFN
from dgm import Net_DGM
import torch.optim as optim
from dgm_pde import DGM_net, Bellman_pde, Train



# Set the batch size and the number of epochs
n_epochs = 100
batch_size = 2

# Set parameters for the LQR equation
H = torch.tensor([[1.0,0],[0,1.0]])
M = torch.tensor([[1.0,0],[0,1.0]])
D = torch.tensor([[0.1,0],[0,0.1]])
C = torch.tensor([[0.1,0],[0,0.1]])
R = torch.tensor([[1.0,0],[0,1.0]])
sigma = torch.tensor([[0.05, 0.05]])

# Create data t and x as defined in exercise 2.1
T = 1.0
x_range = torch.tensor([-3, 3])
y_range = torch.tensor([-3, 3])
t = np.random.uniform(0, T, size=batch_size)
x = np.random.uniform(-3, 3, size=(batch_size, 1, 2))
t0 = torch.from_numpy(np.array([t]).T).float()
x0 = torch.from_numpy(x.reshape(batch_size, 2)).float()
tx = torch.cat([t0,x0], dim=1)

# Convert numpy to torch tensor
t = torch.from_numpy(t)
x = torch.from_numpy(x)

# Determine the value function for the samples of t and x
lqr_equation = LQRSolver(H, M, D, C, R, sigma, T)
opt_control = lqr_equation.optimal_control(t ,x).float()
value_func = lqr_equation.value_function(t, x,).float()

#######################################################################################################
########################################### Algorithm #################################################
#######################################################################################################
 
# Input for FFN neural network (control function)
dim = [3,100,100,2] 
# Input for Net_DGM  neural network (value function)
value_dim_input = 2 
value_dim_hidden = 100

# Input for DGM_Net neural network (PDE)
dim_input = 3
dim_output = 1
num_layers = 3
num_neurons = 50
learning_rate = 0.001

# Initialize the control model, loss function, and Adam optimizer
control_model = FFN(sizes=dim)
# control_loss_fn = nn.MSELoss()
control_optimizer = optim.Adam(control_model.parameters(), lr=learning_rate)

# Initailize the value function model, loss function, and Adam optimizer
value_model = Net_DGM(value_dim_input, value_dim_hidden)
# value_loss_fn = nn.MSELoss()
value_optimizer = optim.Adam(value_model.parameters(), lr=learning_rate)

alpha_pred = control_model(tx)
print(alpha_pred)

# alpha_pred = torch.tensor(np.array([[1.0, 1.0], [1.0, 1.0]])).float()
alpha_pred = alpha_pred.unsqueeze(1).reshape(batch_size,2,1).clone().detach()

print(alpha_pred.type())
net = DGM_net(dim_input, dim_output, num_layers, num_neurons)
Bellman = Bellman_pde(net, x_range, y_range, H, M, C, D, R, T, sigma, alpha_pred)
train = Train(net, Bellman, BATCH_SIZE=batch_size)
train.train(epoch=n_epochs, lr=learning_rate)

# Train the neural networks
# for epoch in range(1):

    # Solve the Bellman PDE using the alpha obtained by the prediction above
    # alpha_pred = alpha_pred.unsqueeze(1).reshape(batch_size,2,1)
    # net = DGM_net(dim_input, dim_output, num_layers, num_neurons)
    # Bellman = Bellman_pde(net, x_range, y_range, H, M, C, D, R, T, sigma, alpha_pred)
    # train = Train(net, Bellman, BATCH_SIZE=batch_size)
    # train.train(epoch=100, lr=learning_rate)
    # pde_value_pred = net(tx)

    # # Update value function based on value obtained from the PDE solution above
    # value_pred = value_model(t0,x0)
    # value_optimizer.zero_grad()
    # value_loss_fn = nn.MSELoss()
    # value_loss = value_loss_fn(value_pred, pde_value_pred)
    # value_loss.backward()
    # value_optimizer.step()

    # # Update the weights for alpha by minimizing the Hamiltonian
    # tx = tx.requires_grad_(True)

    # # 1st order derivatives
    # grad = torch.autograd.grad(net(tx), tx, grad_outputs=torch.ones_like(net(tx)), create_graph=True)
    # du_dt = grad[0][:,0].reshape(-1, 1)  # derivative w.r.t. time, dim = batchsize*1
    # du_dx = grad[0][:,1:] # derivative w.r.t. space, dim = batchsize*2 

    # x_space = tx[:,1:].unsqueeze(1).reshape(batch_size,2,1) # extract (x1,x2)^T, dim = batchsize*2*1
    # x_space_t = x_space.reshape(batch_size,1,2) # dim = batchsize*1*2
    # du_dx_ext_t = du_dx.unsqueeze(1) # dim=batchsize*1*2
    
    # hamiltonian = torch.bmm(du_dx_ext_t,torch.bmm(H_ext,x_space)).squeeze(1)\
    #         +torch.bmm(du_dx_ext_t,torch.bmm(M_ext,alpha_pred)).squeeze(1)\
    #         +torch.bmm(x_space_t,torch.bmm(C_ext,x_space)).squeeze(1)\
    #         +torch.bmm(alpha_pred.reshape(batch_size,1,2),torch.bmm(D_ext,alpha_pred)).squeeze(1) # dim = batchsize*1

    # control_optimizer.zero_grad()
    # control_loss_fn = nn.MSELoss()
    # alpha_loss = control_loss_fn(hamiltonian, torch.zeros_like(hamiltonian))
    # alpha_loss.backward()
    # control_optimizer.step()
    # alpha_pred = control_model(tx)





