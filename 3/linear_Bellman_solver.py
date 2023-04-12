import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from value_simulator_cons_control import LQR_simulator_cons_control

class DGM_layer(nn.Module):
    def __init__(self, in_features, out_feature, residual = False):
        super(DGM_layer, self).__init__()
        self.residual = residual
        
        self.Z = nn.Linear(out_feature, out_feature).double(); self.UZ = nn.Linear(in_features, out_feature, bias = False).double()
        self.G = nn.Linear(out_feature, out_feature).double(); self.UG = nn.Linear(in_features, out_feature, bias = False).double()
        self.R = nn.Linear(out_feature, out_feature).double(); self.UR = nn.Linear(in_features, out_feature, bias = False).double()
        self.H = nn.Linear(out_feature, out_feature).double(); self.UH = nn.Linear(in_features, out_feature, bias = False).double()
        
    def forward(self, x, s):
        z = torch.tanh(self.UZ(x)+self.Z(s)).double()
        g = torch.tanh(self.UG(x)+self.G(s)).double()
        r = torch.tanh(self.UR(x)+self.R(s)).double()
        h = torch.tanh(self.UH(x)+self.H(s)).double()
        return ((1-g)*h+z*s).double()

class DGM_net(nn.Module):
    def __init__(self, in_dim, out_dim, n_layers, n_neurons, residual = False): 
        '''
        in_dim: number of cordinates
        out_dim: number of the output
        make residual = true for identity between each DGM layers 
        '''
        super(DGM_net, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.residual = residual
    
        self.input_layer = nn.Linear(in_dim, n_neurons).double()
        self.dgm_layers = nn.ModuleList([DGM_layer(self.in_dim, self.n_neurons, 
                                               self.residual) for i in range(self.n_layers)])
        self.output_layer = nn.Linear(n_neurons, out_dim).double()
    
    def forward(self, x):
        s = torch.tanh(self.input_layer(x)).double()
        for i, dgm_layer in enumerate(self.dgm_layers):
            s = dgm_layer(x, s).double()
            
        return self.output_layer(s).double()


class Bellman_pde():
    '''
    Approximating the Bellman PDE on [0,T]*[x1_l,x1_r]*[x2_l,x2_r]
    '''
    def __init__(self, net, x1_interval, x2_interval, H, M, C, D, R, T, sigma, a):
        '''
        net: neural network
        x_interval, y_interval: torch tensor, dim=2
        '''
        self.net = net 
        self.x1_l, self.x1_r = x1_interval[0].item(), x1_interval[1].item() 
        self.x2_l, self.x2_r = x2_interval[0].item(), x2_interval[1].item()
        self.H, self.M, self.C, self.D, self.R = H.double(), M.double(), C.double(), D.double(), R.double() # H, M, C, D, R: torch tensors, dim = 2*2      
        self.T = T 
        
        if sigma.shape == torch.Size([1, 2]):
            self.sigma = sigma.t().double()
        elif sigma.shape == torch.Size([2]):
            self.sigma = sigma[:,None].double()
        elif sigma.shape == torch.Size([2,1]):
            self.sigma = sigma.double()
        else:
            raise ValueError('Sigma must be torch tensor with dimension 1*2, 2*1 or 2!')
        
        if a.shape == torch.Size([1, 2]):
            self.a = a.t().double()
        elif a.shape == torch.Size([2]):
            self.a = a[:,None].double()
        elif a.shape == torch.Size([2,1]):
            self.a = sigma.double()
        else:
            raise ValueError('Sigma must be torch tensor with dimension 1*2, 2*1 or 2!')

        
    def sample(self, size):
        # Sampling       
        t_x = torch.cat((torch.rand([size, 1])*self.T, (self.x1_l - self.x1_r) * torch.rand([size, 1]) + self.x1_r, (self.x2_l - self.x2_r) * torch.rand([size, 1]) + self.x2_r), dim=1)
        # samples = [[t,x1,x2],
        #            [t,x1,x2],
        #               ...
        #            [t,x1,x2]]
        
        x_boundary = torch.cat((torch.ones(size, 1)*self.T, (self.x1_l - self.x1_r) * torch.rand([size, 1]) + self.x1_r, (self.x2_l - self.x2_r) * torch.rand([size, 1]) + self.x2_r), dim=1)
        # samples = [[T,x1,x2],
        #            [T,x1,x2],
        #              ...
        #            [T,x1,x2]]
        
        return t_x.double(), x_boundary.double() # dim = batchsize*3

        
    def get_hes_space(self, grad_x, x):
        '''
        Compute the Hessian matrix [[d^2u/dx^2, d^2u/dxdy],
                                    [d^2u/dydx, d^2u/dy^2]]
        '''
        hessian = torch.zeros(len(x),2,2)
        dxx = torch.autograd.grad(grad_x[0][:,1], x, grad_outputs=torch.ones_like(grad_x[0][:,1]), allow_unused=True, retain_graph=True)[0][:,1]
        dxy = torch.autograd.grad(grad_x[0][:,1], x, grad_outputs=torch.ones_like(grad_x[0][:,1]), allow_unused=True, retain_graph=True)[0][:,2]
        dyx = torch.autograd.grad(grad_x[0][:,2], x, grad_outputs=torch.ones_like(grad_x[0][:,2]), allow_unused=True, retain_graph=True)[0][:,1]
        dyy = torch.autograd.grad(grad_x[0][:,2], x, grad_outputs=torch.ones_like(grad_x[0][:,2]), allow_unused=True, retain_graph=True)[0][:,2]
        hessian[:,0,0] = dxx 
        hessian[:,0,1] = dxy
        hessian[:,1,0] = dyx
        hessian[:,1,1] = dyy

        return hessian.double()  # dim = batchsize*2*2
        
        
    def loss_func(self, size):
        
        loss = nn.MSELoss() # Function for computing the MSE loss
        
        H, M, C, D, R = self.H, self.M, self.C, self.D, self.R
        sig, a = self.sigma, self.a
           
        x, x_boundary = self.sample(size=size)
        x = x.requires_grad_(True) # Track gradients during automatic differentiation

        # 1st order derivatives
        grad = torch.autograd.grad(self.net(x), x, grad_outputs=torch.ones_like(self.net(x)), create_graph=True)
        
        du_dt = grad[0][:,0].reshape(-1, 1)  # derivative w.r.t. time, dim = batchsize*1
        # du/dt = [[u_t]
        #          [u_t]
        #           ...
        #          [u_t]]  
        
        du_dx = grad[0][:,1:] # derivative w.r.t. space, dim = batchsize*2 
        # du/dx = [[u_x, u_y]
        #          [u_x, u_y]
        #              ...
        #          [u_x, u_y]] 

        hessian = self.get_hes_space(grad,x) # dim = batchsize*2*2
        
        # Error from the pde
        trace = torch.diagonal(sig@sig.t()@hessian, dim1=1,dim2=2).sum(dim=1)[None,:].t() # dim = batchsize*1
        x_space = x[:,1:].unsqueeze(1).reshape(size,2,1) # extract (x1,x2)^T, dim = batchsize*2*1
        x_space_t = x_space.reshape(size,1,2) # dim = batchsize*1*2
        du_dx_ext_t = du_dx.unsqueeze(1) # dim=batchsize*1*2
        
        pde = du_dt+0.5*trace+(du_dx_ext_t@H@x_space).squeeze(1)+(du_dx_ext_t@M@a).squeeze(1)\
              +(x_space_t@C@x_space).squeeze(1)+(a.t()@D@a).squeeze(1)
        
        pde_err = loss(pde, torch.zeros(size,1,dtype=torch.float64))
        
        # Error from the boundary condition
        x_bound = x_boundary[:,1:].unsqueeze(1).reshape(size,2,1) # extract (x1,x2)^T, dim = batchsize*2*1
        x_bound_t = x_bound.reshape(size,1,2) # dim = batchsize*1*2
        
        boundary_err = loss(self.net(x_boundary), torch.bmm(x_bound_t,torch.bmm(R[None,:].repeat(size,1,1),x_bound)).squeeze(1))
        
        return pde_err + boundary_err


class Train():
    def __init__(self, net, PDE, BATCH_SIZE):
        '''
        data_in: dim=batchsize*3
        '''
        self.errors = []
        self.err_sim = []
        self.BATCH_SIZE = BATCH_SIZE
        self.net = net
        self.model = PDE

    def train(self, epoch, lr, data = None):
        optimizer = optim.Adam(self.net.parameters(), lr) # Import the parameters, lr: learning rate
        avg_loss = 0
        if data == None:
            for e in range(epoch):
                optimizer.zero_grad()
                loss = self.model.loss_func(self.BATCH_SIZE)
                avg_loss = avg_loss + loss.item()
                loss.backward()
                optimizer.step()
                if e % 50 == 49:
                    loss = avg_loss/50
                    print("epoch {} - lr {} - loss: {}".format(e, lr, loss))
                    avg_loss = 0
                    error = self.model.loss_func(self.BATCH_SIZE)
                    self.errors.append(error.detach())
        else:
            simulator = LQR_simulator_cons_control(self.model.T, self.model.H, self.model.M, self.model.C, self.model.D, self.model.R, self.model.sigma)
            t = data[:,0]
            xx = data[:,1:].unsqueeze(1)
            val_sim = simulator.val_sim(t, xx, 1000, 5000)
            err_sim = self.err_sim
            l1_err = 0
            for e in range(epoch):
                optimizer.zero_grad()
                loss = self.model.loss_func(self.BATCH_SIZE)
                avg_loss += float(loss.item())
                loss.backward()
                optimizer.step()
                if e % 50 == 49:
                    loss = avg_loss/50
                    l1_err = 1/len(data)*torch.sum(torch.abs(self.net(data)-val_sim)).item()
                    l1_err = float(l1_err)
                    print("epoch {} - lr {} - loss: {} - L1 error at epoch {}: {}".format(e, lr, loss, e, l1_err))
                    avg_loss = 0
                    error = self.model.loss_func(self.BATCH_SIZE)
                    self.errors.append(float(error.detach()))
                    err_sim.append(l1_err)
                    l1_err = 0

    def get_errors(self):
        return self.errors, self.err_sim