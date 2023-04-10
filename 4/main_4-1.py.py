import numpy as np
import torch
import torch.nn as nn
from linear_quadratic_regulator import LQRSolver
from functions import FFN
from dgm import Net_DGM
import torch.optim as optim
from dgm_pde import DGM_net, Bellman_pde, Train
import matplotlib.pyplot as plt

def mat_ext(mat, size):
        if mat.shape == torch.Size([2, 2]):
            return mat.unsqueeze(0).repeat(size,1,1)
        elif mat.shape == torch.Size([1, 2]):
            return mat.t().unsqueeze(0).repeat(size,1,1)

# Set the batch size and the number of epochs
n_epochs = 100
batch_size = 10

# Set parameters for the LQR equation
H = torch.tensor([[1.0,0],[0,1.0]])
M = torch.tensor([[1.0,0],[0,1.0]])
D = torch.tensor([[0.1,0],[0,0.1]])
C = torch.tensor([[0.1,0],[0,0.1]])
R = torch.tensor([[1.0,0],[0,1.0]])
sigma = torch.tensor([[0.05, 0.05]])

H_ext = mat_ext(H, batch_size)
M_ext = mat_ext(M, batch_size)
D_ext = mat_ext(D, batch_size)
C_ext = mat_ext(C, batch_size)
R_ext = mat_ext(R, batch_size)
sigma = torch.tensor([[0.05, 0.05]])

# Create data t and x as defined in exercise 2.1
T = 1.0

# Determine the value function for the samples of t and x


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

# Initialize the control model and Adam optimizer
control_model = FFN(sizes=dim)
control_optimizer = optim.Adam(control_model.parameters(), lr=learning_rate)

# Initailize the value function model and Adam optimizer
value_model = Net_DGM(value_dim_input, value_dim_hidden)
value_optimizer = optim.Adam(value_model.parameters(), lr=learning_rate)

net = DGM_net(dim_input, dim_output, num_layers, num_neurons)

print_counter = 0

v_loss = torch.zeros(100)
a_loss = torch.zeros(100)

x_range = torch.tensor([-3, 3])
y_range = torch.tensor([-3, 3])

# Train the neural networks
for epoch in range(100):
    t = np.random.uniform(0, T, size=batch_size)
    x = np.random.uniform(-3, 3, size=(batch_size, 1, 2))
    t0 = torch.from_numpy(np.array([t]).T).float()
    t0 = t0.requires_grad_(True)
    x0 = torch.from_numpy(x.reshape(batch_size, 2)).float()
    x0 = x0.requires_grad_(True)
    tx = torch.cat([t0,x0], dim=1)

    # Convert numpy to torch tensor
    t = torch.from_numpy(t)
    x = torch.from_numpy(x)

    alpha_pred = control_model(tx)
    pde_value_pred = torch.zeros((batch_size,1))

    for i in range(batch_size):
        # Solve the Bellman PDE using the alpha obtained by the prediction above
        alpha_pred_i = alpha_pred[i].unsqueeze(1).reshape(1,2,1)
        Bellman = Bellman_pde(net, x_range, y_range, H, M, C, D, R, T, sigma, alpha_pred_i)
        train = Train(net, Bellman, BATCH_SIZE=1)
        train.train(epoch=50, lr=learning_rate)
        # Might also be problem with tx dimensions here
        pde_value_pred[i][0] = net(tx[i])

    pde_value_pred = pde_value_pred.detach().clone()

    # Update value function based on value obtained from the PDE solution above
    value_model.zero_grad()
    value_loss_fn = nn.MSELoss()
    value_pred = value_model(t0,x0)
    value_loss = value_loss_fn(value_pred, pde_value_pred)
    value_loss.backward(retain_graph=True)
    value_optimizer.step()
    new_value_pred = value_model(t0,x0)

    v_loss[epoch] = value_loss

    # Update the weights for alpha by minimizing the Hamiltonian
    tx = tx.requires_grad_(True)

    # 1st order derivatives
    grad_x = torch.autograd.grad(new_value_pred, x0, grad_outputs=torch.ones_like(new_value_pred), create_graph=True)
    du_dx = grad_x[0]  # derivative w.r.t. time, dim = batchsize*1

    x_space = tx[:,1:].unsqueeze(1).reshape(batch_size,2,1) # extract (x1,x2)^T, dim = batchsize*2*1
    x_space_t = x_space.reshape(batch_size,1,2) # dim = batchsize*1*2
    du_dx_ext_t = du_dx.unsqueeze(1) # dim=batchsize*1*2

    alpha_pred = alpha_pred.unsqueeze(1).reshape(batch_size,2,1)
    
    hamiltonian = torch.bmm(du_dx_ext_t,torch.bmm(H_ext,x_space)).squeeze(1)\
            +torch.bmm(du_dx_ext_t,torch.bmm(M_ext,alpha_pred)).squeeze(1)\
            +torch.bmm(x_space_t,torch.bmm(C_ext,x_space)).squeeze(1)\
            +torch.bmm(alpha_pred.reshape(batch_size,1,2),torch.bmm(D_ext,alpha_pred)).squeeze(1) # dim = batchsize*1

    control_loss_fn = nn.L1Loss()
    control_model.zero_grad()
    alpha_loss = control_loss_fn(hamiltonian, torch.zeros_like(hamiltonian))
    alpha_loss.backward(retain_graph=True)
    control_optimizer.step()

    a_loss[epoch] = alpha_loss

    if (print_counter == 10):
        print(f"Iteration: {epoch}")
        print(f"Value function loss: {value_loss}")
        print(f"Control Loss: {alpha_loss}")
        print()
        print_counter = 0
    print_counter += 1

t = np.array([0.1, 0.5])
x = np.array([[[0.8, 1]],
              [[2, -1]]])

t0 = torch.from_numpy(np.array([t]).T).float()
x0 = torch.from_numpy(x.reshape(2, 2)).float()
tx = torch.cat([t0,x0], dim=1)

t = torch.from_numpy(t)
x = torch.from_numpy(x)

value_prediction = value_model(t0,x0)
control_prediction = control_model(tx)

lqr_equation = LQRSolver(H, M, D, C, R, sigma, T)
control_actual = lqr_equation.optimal_control(t ,x).float()
value_actual = lqr_equation.value_function(t, x,).float()

# Determine MSE loss between prediction and true values
loss_func = torch.nn.MSELoss()
print(f"MSE of control alpha: {loss_func(control_prediction, control_actual)}")
print(f"MSE of control alpha: {loss_func(value_prediction, value_actual)}")
print()

print(control_actual)
print(control_prediction)
print(value_actual)
print(value_prediction)

v_loss = v_loss.detach().numpy()
a_loss = a_loss.detach().numpy()

fig, axes = plt.subplots(2, figsize=(8, 7))

axes[0].plot(v_loss)
axes[0].set_title(f"Value Loss")
axes[0].set_xlabel("Number of Epochs")
axes[0].set_ylabel("Loss")

axes[1].plot(a_loss)
axes[1].set_title(f"Alpha Loss")
axes[1].set_xlabel("Number of Epochs")
axes[1].set_ylabel("Loss")

fig.tight_layout()
plt.show()




