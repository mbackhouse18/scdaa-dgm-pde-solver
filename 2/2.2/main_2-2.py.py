import torch
import torch.nn as nn
from functions import FFN
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
n_epochs = 150
batch_size = 30

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
opt_control = lqr_equation.optimal_control(t ,x).float()

tx = torch.cat([t0,x0], dim=1)

# Define the dimensions of the input and the dimension of the hidden layer
# This will produce two hidden layers each of size 100
dim = [3,100,100,2]


######################## Mean-Squared Error Loss Function ########################

# Define model, loss function, and Adam optimizer
model = FFN(sizes=dim)
loss_fn = nn.MSELoss()  # Mean-Squared Error
optimizer = optim.Adam(model.parameters(), lr=0.001)

loss_list_L1 = torch.zeros(n_epochs)

# Train the neural network
for epoch in range(n_epochs):
    y_pred = model(tx)
    loss = loss_fn(y_pred, opt_control)
    loss_list_L1[epoch] = loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

loss_list = loss_list_L1.detach().numpy()
prediction = model(tx).detach().numpy()
true_con = opt_control.detach().numpy()
x_axis = np.linspace(0,batch_size, num=batch_size)

fig, axes = plt.subplots(3, figsize=(10, 7))

# Plot the loss
axes[0].plot(loss_list, label='Mean-Squared Error')
axes[0].set_title("Mean-Squared Error Loss Function")
axes[0].set_xlabel("Number of Epochs")
axes[0].set_ylabel("Loss")

# Plot the predicted vs true values for each dimension of the control
for i in range(len(prediction[0])):
    axes[i+1].scatter(x_axis, prediction[:,i], label='Predicted')
    axes[i+1].scatter(x_axis, true_con[:,i], label='True')
    axes[i+1].set_title(f"Dimension {i+1}: Predicted vs. True Value")
    axes[i+1].set_xlabel("Batch")
    axes[i+1].set_ylabel("Value")
    axes[i+1].legend(loc='upper right')

fig.tight_layout()
plt.show()