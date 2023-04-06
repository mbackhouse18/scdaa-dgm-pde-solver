# Libraries
import torch 
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim

class DGM_layer(nn.Module):
    def __init__(self, in_features, out_feature, residual = False):
        super(DGM_layer, self).__init__()
        self.residual = residual
        
        self.Z = nn.Linear(out_feature, out_feature); self.UZ = nn.Linear(in_features, out_feature, bias = False)
        self.G = nn.Linear(out_feature, out_feature); self.UG = nn.Linear(in_features, out_feature, bias = False)
        self.R = nn.Linear(out_feature, out_feature); self.UR = nn.Linear(in_features, out_feature, bias = False)
        self.H = nn.Linear(out_feature, out_feature); self.UH = nn.Linear(in_features, out_feature, bias = False)
        
    def forward(self, x, s):
        z = torch.tanh(self.UZ(x)+self.Z(s))
        g = torch.tanh(self.UG(x)+self.G(s))
        r = torch.tanh(self.UR(x)+self.R(s))
        h = torch.tanh(self.UH(x)+self.H(s))
        return (1-g)*h+z*s  

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
    
        self.input_layer = nn.Linear(in_dim, n_neurons)
        self.dgm_layers = nn.ModuleList([DGM_layer(self.in_dim, self.n_neurons, 
                                               self.residual) for i in range(self.n_layers)])
        self.output_layer = nn.Linear(n_neurons, out_dim)
    
    def forward(self, x):
        s = torch.tanh(self.input_layer(x))
        for i, dgm_layer in enumerate(self.dgm_layers):
            s = dgm_layer(x, s)
            
        return self.output_layer(s)


class Bellman_pde():
    '''
    Approximating the Bellman PDE on [0,T]*[x1_l,x1_r]*[x2_l,x2_r]
    '''
    def __init__(self, net, x_interval, y_interval, H, M, C, D, R, T, sigma, a):
        self.net = net 
        self.x1_l = x_interval[0].item() # torch tensor, dim = 3
        self.x1_r = x_interval[1].item()
        self.x2_l = y_interval[0].item()
        self.x2_r = y_interval[1].item()
        self.H = H # H, M, C, D, R: torch tensors, dim = 2*2
        self.M = M 
        self.C = C 
        self.D = D 
        self.R = R         
        self.T = T # integer
        self.sigma = sigma # sigma, a: torch tensors, dim = 1*2
        self.a = a 
        
    def sample(self, size):
        # Sampling
        #te  = self.ts[0].item()    
        #x1e = self.ts[1].item()
        #x2e = self.ts[2].item()
        
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
        
        return t_x, x_boundary
    
    def mat_ext(self, mat, size):
        if mat.shape == torch.Size([2, 2]):
            return mat.unsqueeze(0).repeat(size,1,1)
        elif mat.shape == torch.Size([1, 2]):
            return mat.t().unsqueeze(0).repeat(size,1,1)
        
    def get_hes_space(self, grad_x, x):
        hessian = torch.zeros(len(x),2,2)
        dxx = torch.autograd.grad(grad_x[0][:,1], x, grad_outputs=torch.ones_like(grad_x[0][:,1]), allow_unused=True, retain_graph=True)[0][:,1]
        dxy = torch.autograd.grad(grad_x[0][:,1], x, grad_outputs=torch.ones_like(grad_x[0][:,1]), allow_unused=True, retain_graph=True)[0][:,2]
        dyx = torch.autograd.grad(grad_x[0][:,2], x, grad_outputs=torch.ones_like(grad_x[0][:,2]), allow_unused=True, retain_graph=True)[0][:,1]
        dyy = torch.autograd.grad(grad_x[0][:,2], x, grad_outputs=torch.ones_like(grad_x[0][:,2]), allow_unused=True, retain_graph=True)[0][:,2]
        hessian[:,0,0] = dxx 
        hessian[:,0,1] = dxy
        hessian[:,1,0] = dyx
        hessian[:,1,1] = dyy
        return hessian  
        
        
    def loss_func(self, size):
        
        loss = nn.MSELoss() # MSE 
        
        # Extend the input matrices
        H = self.mat_ext(self.H, size) # H, M, C, D, R: dim = batchsize*2*2
        M = self.mat_ext(self.M, size)
        C = self.mat_ext(self.C, size)
        D = self.mat_ext(self.D, size)
        R = self.mat_ext(self.R, size) # control: dim = batchsize*2*1          
        T = self.T
        a = self.a
        sig = self.sigma.t() # dim = 2*1        

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
                
        # Hessian matrix
        hessian = self.get_hes_space(grad,x)
        
        # Error from the pde
        sig2_ext = self.mat_ext(torch.matmul(sig,sig.t()), size) # dim = batchsize*2*2
        prod = torch.bmm(sig2_ext,hessian) # sigma*sigma^T*2nd derivatives
        trace = torch.diagonal(prod, dim1=1, dim2=2).sum(dim=1).unsqueeze(0).t() # trace, dim = batchsize*1
        x_space = x[:,1:].unsqueeze(1).reshape(size,2,1) # extract (x1,x2)^T, dim = batchsize*2*1
        x_space_t = x_space.reshape(size,1,2) # dim = batchsize*1*2
        du_dx_ext_t = du_dx.unsqueeze(1) # dim=batchsize*1*2
        
        pde = du_dt+0.5*trace+torch.bmm(du_dx_ext_t,torch.bmm(H,x_space)).squeeze(1)\
                +torch.bmm(du_dx_ext_t,torch.bmm(M,self.a)).squeeze(1)\
                +torch.bmm(x_space_t,torch.bmm(C,x_space)).squeeze(1)\
                +torch.bmm(a.reshape(size,1,2),torch.bmm(D,a)).squeeze(1) # dim = batchsize*1
 
        pde_err = loss(pde, torch.zeros(size,1))
        
        # Error from the boundary condition
        x_bound = x_boundary[:,1:].unsqueeze(1).reshape(size,2,1) # extract (x1,x2)^T, dim = batchsize*2*1
        x_bound_t = x_bound.reshape(size,1,2) # dim = batchsize*1*2
        # boundary = self.net(x_boundary)-torch.bmm(x_bound_t,torch.bmm(R,x_bound)).squeeze(1) # dim = batchsize*1
        
        boundary_err = loss(self.net(x_boundary), torch.bmm(x_bound_t,torch.bmm(R,x_bound)).squeeze(1))
        
        return pde_err + boundary_err


class Train():
    def __init__(self, net, PDE, BATCH_SIZE):
        self.errors = []
        self.BATCH_SIZE = BATCH_SIZE
        self.net = net
        self.model = PDE

    def train(self, epoch, lr):
        optimizer = optim.Adam(self.net.parameters(), lr) # Import the parameters, lr: learning rate
        avg_loss = 0
        for e in range(epoch):
            optimizer.zero_grad()
            loss = self.model.loss_func(self.BATCH_SIZE)
            avg_loss = avg_loss + float(loss.item())
            loss.backward()
            optimizer.step()
            if e % 100 == 99:
                loss = avg_loss/100
                print("epoch {} - lr {} - loss: {}".format(e, lr, loss))
                avg_loss = 0

                error = self.model.loss_func(self.BATCH_SIZE)
                self.errors.append(error.detach())

    def get_errors(self):
        return self.errors