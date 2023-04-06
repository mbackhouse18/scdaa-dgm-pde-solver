import torch
import torch.nn as nn
from dgm import Net_DGM
import numpy as np
from linear_quadratic_regulator import LQRSolver
import torch.optim as optim
import matplotlib.pyplot as plt

# Set parameters for the LQR equation
H = torch.tensor([[1.0,0],[0,1.0]])
M = torch.tensor([[1.0,0],[0,1.0]])
D = torch.tensor([[0.1,0],[0,0.1]])
C = torch.tensor([[0.1,0],[0,0.1]])
R = torch.tensor([[1.0,0],[0,1.0]])
sigma = torch.tensor([0.05, 0.05])

# Set the batch size and the number of epochs
n_epochs = 200
batch_size = 10000

# Create data t and x as defined in exercise 2.1
T = 1.0
t = np.random.uniform(0, T, size=batch_size)
x = np.random.uniform(-3, 3, size=(batch_size, 1, 2))

t0 = torch.from_numpy(np.array([t]).T).float()
x0 = torch.from_numpy(x.reshape(batch_size, 2)).float()

# Convert numpy to torch tensor
t = torch.from_numpy(t)
x = torch.from_numpy(x)

# Determine the value function for the samples of t and x
lqr_equation = LQRSolver(H, M, D, C, R, sigma, T)
value_funs = lqr_equation.value_function(t ,x).float()

# Define the dimensions of the input and the dimension of the hidden layer
dim_input = 2
dim_hidden = 100



######################## Mean-Squared Error Loss Function ########################

# Define model, loss function, and Adam optimizer
model = Net_DGM(dim_input, dim_hidden)
loss_fn = nn.MSELoss()  # Mean-Squared Error
optimizer = optim.Adam(model.parameters(), lr=0.001)

loss_list_L1 = torch.zeros(n_epochs)

# Train the neural network
for epoch in range(n_epochs):
    y_pred = model(t0,x0)
    loss = loss_fn(y_pred, value_funs)
    loss_list_L1[epoch] = loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

loss_list = loss_list_L1.detach().numpy()
prediction = model(t0,x0).detach().numpy()
true_v = value_funs.detach().numpy()
x_axis = np.linspace(0,batch_size, num=batch_size)

fig, axes = plt.subplots(2, figsize=(8, 7))

# Plot the loss
axes[0].plot(loss_list, label='Mean-Squared Error')
axes[0].set_title("Mean-Squared Error Loss Function")
axes[0].set_xlabel("Number of Epochs")
axes[0].set_ylabel("Loss")

# Plot the true and predicted values
axes[1].scatter(x_axis, prediction, label='Predicted')
axes[1].scatter(x_axis, true_v, label='True')
axes[1].set_title("Predicted vs. True Value")
axes[1].set_xlabel("Batch")
axes[1].set_ylabel("Value")
axes[1].legend()

fig.tight_layout()
plt.show()
