""" Inverse Problem for 1-D Euler Equations for Compressible Flow -  The Sod Problem -- Total Energy Excluded
AMSC 663 / 664 -- Advanced Scientific Computing I & II
Author Alexandros D.L Papados
In this code we solve for the original heat capacity ratio, gamma = 1.4, from a full numerical solution data set to
the follow equation:
                               U_t + F_x = 0,         (x,t) in (0,1)x(0,0.2]
                              (rho,u,p)_t=0 = (1.0,0.0,1.0)    0 <= x < 0.5
                              (rho,u,p)_t=0 = (0.125,0.0,0.1) 0.5 < x <=1
 with Dirichlet boundary conditions which take the values of the initial condition at the boundaries
                               U = [ rho ]       and       F  =  [   rho*u      ]
                                   [rho*u]                       [ rho*u^2 + p  ]
                                   [rho E]                       [ u(rho*E + p) ]
rho -- Density of the fluid
u   -- Velocity of the fluid - x direction
p   -- Pressure of the fluid
E   --  Total energy of fluid
We relate the pressure and energy by the equation of state of the form
                                            p = (gamma - 1) ( rho*E - 0.5*rho||u||^2)
The Neural Network is constructed as follows:
                                                         ( sigma(t,x,theta) )
                                                                                                       ( rho(t,x) )
                       ( t )                             ( sigma(t,x,theta) )                          (          )
             Input:            ----> Activation Layers:          .               ----> Output Layer:   (  u(t,x)  )
                       ( x )                                     .                                     (          )
                                                                 .                                     (  p(x,t)  )
                                                         ( sigma(t,x,theta) )
Remark: theta are the weights of the network
The updated heat capacity ratio is printed for each training iteration
For this problem we wish to solve for gamma = 1.4
How to use this code:
Using the function PINNs(X, x_i, x_l, x_r, rho_i, u_i, p_i, rho, u, p, layers) - we set up the network
# X - (x,t) associated to interior points
# x_l - (x,t) associated to left boundary
# x_r - (x,t) associated to right boundary
# rho_i - Exact initial condition for density
# u_i  - Exact initial condition for velocity
# p_i  - Exact initial condition for pressure
# rho - Exact solution for density
# u  - Exact solution for velocity
# p  - Exact solution for pressure
# layers - [ inputs, number of neurons per layer, outputs]   -- ex. [(x,t), 20,20,20, [rho,u,p]]
# We register the parameter gamma so that for each iteration, gamma is updated in accordance to the loss function
# Define the model by model = PINNs(X, x_i, x_l, x_r, rho_i, u_i, p_i, rho, u, p, layers)
# model.train(iter) trains the network and outputs approximated gamma
"""
import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
import time
import scipy.io


torch.manual_seed(123456)                                                          # Seed for torch
np.random.seed(123456)                                                             # Seed for np
q = np.round(np.random.uniform(1.08, 1.66),1)                                      # Random point from distribution
device = torch.device('cpu')

# Add boundary condition loss function to general loss function: Enter 'Yes' or 'No'
BC = 'Yes'

#  Deep Neural Network
class DNN(nn.Module):
    def __init__(self, layers):

        super(DNN, self).__init__()
        self.depth = len(layers) - 1
        self.activation = nn.Tanh                                                  # Activation function for each layer
        layer_list = list()

        for i in range(self.depth - 1):
            layer_list.append(('Linear_Layer_%d'% i, nn.Linear(layers[i], layers[i + 1])))      # Linear layer
            layer_list.append(('Tanh_Layer_%d' % i, self.activation()))                         # Activation layer

        layer_list.append(('Layer_%d' % (self.depth - 1), nn.Linear(layers[-2], layers[-1])))   # Append solution layer to list
        layerDict = OrderedDict(layer_list)                                                     # Recalls the order of entries
        self.layers = nn.Sequential(layerDict)                                                  # Sequential container

    # Forward pass of the network to predict y
    def forward(self, x):
        out = self.layers(x)
        return out

# Physics Informed Neural Network
class PINNs():
    def __init__(self, X, x_i, x_l, x_r, rho_i, u_i, p_i, rho, u, p, layers):


        self.xl = torch.tensor(x_l[:,0:1]).float().to(device)                                   # Left Boundary x
        self.tl = torch.tensor(x_l[:,1:2]).float().to(device)                                   # Left Boundary t
        self.xr = torch.tensor(x_r[:, 0:1]).float().to(device)                                  # Right Boundary x
        self.tr = torch.tensor(x_r[:, 1:2]).float().to(device)                                  # Right Boundary t


        self.x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)                 # Interior x
        self.t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)                 # Interior t

        self.x_i = torch.tensor(x_i[:, 0:1], requires_grad=True).float().to(device)             # x for IC
        self.t_i = torch.tensor(x_i[:, 1:2], requires_grad=True).float().to(device)             # t for IC

        self.rho = torch.tensor(rho).float().to(device)                                         # Exact rho
        self.u = torch.tensor(u).float().to(device)                                             # Exact u
        self.p = torch.tensor(p).float().to(device)                                             # Exact p

        self.rho_i = torch.tensor(rho_i).float().to(device)                                     # Exact rho IC
        self.u_i = torch.tensor(u_i).float().to(device)                                         # Exact u IC
        self.p_i = torch.tensor(p_i).float().to(device)                                         # Exact p IC

        self.gamma = torch.tensor([q], requires_grad=True).to(device)                           # Define gamma

        # Register gamma as parameter to optimize
        self.gamma = nn.Parameter(self.gamma)                                                   # Register gamma
        self.dnn = DNN(layers).to(device)                                                       # DNN
        self.dnn.register_parameter('gamma', self.gamma)                                        # Allow DNN to optimize gamma


        # Optimizer - Limited Memory Broyden–Fletcher–Goldfarb–Shannon Algorithm
        self.optimizer = torch.optim.LBFGS(
            self.dnn.parameters(),                       # Optimize theta and gamma
            lr=1,                                        # Learning Rate
            max_iter = 10000,                            # Default max # of iterations per optimization step                                                                    #
            tolerance_grad = 1e-9,                       # Termination tolerance on first order optimality
            tolerance_change = 1e-9,                     # Termination tolerance on function value/parameter change
            history_size = 5000
        )
        self.iter = 0                                     # Initialize iterations

    # Neural network solution y = [rho(x,t) , u(x,t), p(x,t)]
    def net_y(self, x, t):
        y = self.dnn(torch.cat([x, t], dim=1))
        return y

    # PDE Loss Function
    def loss_pde(self, x, t):
        gamma = self.gamma                                                           # Heat Capacity Ratio
        y = self.net_y(x, t)                                                         # NN(x,t,gamma,theta)
        rho, u, p = y[:, 0], y[:, 1], y[:, 2]                                        # NN_{rho}, NN_{u}, NN_{p}
        E = p / ((gamma - 1) / rho) + 0.5 * u * u                                    # Total Energy

        # Derivatives for each of the physical quantities
        # rho_t
        rho_t = torch.autograd.grad(
            rho, t,
            grad_outputs=torch.ones_like(rho),
            retain_graph=True,
            create_graph=True
        )[0]

        # (rho*u)_t
        rhou_t = torch.autograd.grad(
            rho*u, t,
            grad_outputs=torch.ones_like(rho*u),
            retain_graph=True,
            create_graph=True
        )[0]

        # (rho*u)_x
        rhou_x = torch.autograd.grad(
            rho * u, x,
            grad_outputs=torch.ones_like(rho*u),
            retain_graph=True,
            create_graph=True
        )[0]

        # (rho*u^2 + p)_x
        rhoup_x = torch.autograd.grad(
            (rho * u * u) + p, x,
            grad_outputs=torch.ones_like((rho * u * u) + p),
            retain_graph=True,
            create_graph=True
        )[0]

        # (rho*E)_t
        rhoE_t = torch.autograd.grad(
            rho*E, t,
            grad_outputs=torch.ones_like(rho*E),
            retain_graph=True,
            create_graph=True
        )[0]

        # (u * (rho * E + p))_x
        rhoupE_x = torch.autograd.grad(
            u * (rho * E + p), x,
            grad_outputs=torch.ones_like( u * (rho * E + p)),
            retain_graph=True,
            create_graph=True
        )[0]

        # Loss function for (U_nn_t + grad(F_nn)) = 0
        loss_euler = ((rho_t + rhou_x) ** 2).mean() + \
                     ((rhou_t + rhoup_x) ** 2).mean() + \
                     ((rhoE_t + rhoupE_x) ** 2).mean()

        return loss_euler

    # Boundary Condition Loss Function
    def loss_bc(self,xl,tl,xr,tr):
        y_l, y_r = self.net_y(xl,tl), self.net_y(xr,tr)                         # NN left and right boundary
        rho_l, u_l, p_l, = y_l[:, 0], y_l[:, 1], y_l[:, 2]
        rho_r, u_r, p_r = y_r[:, 0], y_r[:, 1], y_r[:, 2]

        # Loss of BCs
        loss_bcs = ((rho_l - 1.0) ** 2).mean() + ((rho_r - 0.125) ** 2).mean() + \
                ((u_l - 0.0) ** 2).mean() + ((u_r - 0.0) ** 2).mean() + \
                ((p_l - 1.0) ** 2).mean() + ((p_r - 0.1) ** 2).mean()
        return loss_bcs

    def loss_ic(self,rho_i,u_i,p_i):
        y_pred = self.net_y(self.x_i,self.t_i)                                  # NN initial condition
        rho_i_pred, u_i_pred, p_i_pred = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]

        # Loss of ICs
        loss_ics = ((u_i_pred - u_i) ** 2).mean() + \
                ((rho_i_pred - rho_i) ** 2).mean() + \
                ((p_i_pred - p_i) ** 2).mean()

        return  loss_ics

    # General Loss Function
    def loss_func(self):
        y = self.net_y(self.x, self.t)
        rho_pred,u_pred, p_pred = y[:,0],y[:, 1],y[:, 2]
        f_pred = self.loss_pde(self.x, self.t)
        loss_ics = self.loss_ic(self.rho_i, self.u_i, self.p_i)
        loss_bcs = self.loss_bc(self.xl, self.tl,self.xr,self.tr)
        # Weighted Total Loss
        if BC == 'Yes':
            loss =  0.01*(((rho_pred - self.rho)**2).mean() + ((u_pred - self.u)**2).mean() + \
                 ((p_pred - self.p) ** 2).mean() + f_pred + 0.01*loss_ics + 0.0025*loss_bcs)
        elif BC == 'No':
            loss =  0.01*(((rho_pred - self.rho) ** 2).mean() + ((u_pred - self.u) ** 2).mean() + \
                           ((p_pred - self.p) ** 2).mean() + f_pred + 0.001*loss_ics )

        # Minimize loss
        self.optimizer.zero_grad()
        loss.backward()

        self.iter += 1

        print(
                'Iteration %d, Loss: %e,  gamma_exact: %.5f , gamma_PINNs: %.5f,' %
                (
                    self.iter,
                    loss.item(),
                    1.40,
                    self.gamma.item()
                )
            )
        return loss

    # Train network through minimization of loss function w/r to theta and gamma
    def train(self,iter):
        self.dnn.train()
        # Update parameter
        self.optimizer.step(self.loss_func)

# Initial Conditions
def IC(x):
    N = len(x)
    rho_init = np.zeros((x.shape[0]))
    u_init = 0.0*np.ones((x.shape[0]))
    p_init = np.zeros((x.shape[0]))
    for i in range(N):
        if (x[i] <= 0.5):
            rho_init[i] = 1.0
        else:
            rho_init[i] = 0.125

    for i in range(N):
        if (x[i] <=0.5):
            p_init[i] = 1.0
        else:
            p_init[i] = 0.1
    return rho_init, u_init, p_init

# Initialization
layers = [2, 70, 70, 70, 3]                                                # [Input, Neurons per Layer, Output]
data = scipy.io.loadmat('Euler_Sod.mat')                                   # Import Solution data
t = data['t'].flatten()[:,None]                                            # Partitioned time coordinates
x = data['x'].flatten()[:,None]                                            # Partitioned spatial coordinates

num_x = len(x)                                                             # Length of x array
num_t = len(t)                                                             # Length of t array

num_i_train = 1000                                                         # Sampling from IC
num_b_train = 100                                                          # Sampling from Boundary
num_f_train = 3975                                                         # Sampling from Interior

Exact_rho = np.real(data['r_full'])                                        # Exact density
Exact_u = np.real(data['u_full'])                                          # Exact velocity
Exact_p = np.real(data['p_full'])                                          # Exact pressure


x_grid, t_grid = np.meshgrid(x,t)
T = t_grid.flatten()[:, None]
X = x_grid.flatten()[:, None]
X_full = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))             # Each combination of (x,t)

id_i = np.random.choice(num_x, num_i_train, replace=False)                 # Randomly chosen points for IC
id_b = np.random.choice(num_t, num_b_train, replace=False)                 # Randomly chosen points for BC
id_f = np.random.choice(num_x*num_t, num_f_train, replace=False)           # Randomly chosen points for Interior

# Vectorized exact solutions
rho_exact = Exact_rho.flatten()[:,None]
u_exact = Exact_u.flatten()[:,None]
p_exact = Exact_p.flatten()[:,None]

# Obtain random points from grid for IC
x_ic = x_grid[0,id_i][:, None]                                             # x of IC
t_ic = t_grid[0,id_i][:, None]                                             # t of IC
x_ic_train = np.hstack((x_ic,t_ic))                                        # Combinations of these points
rho_ic, u_ic, p_ic = IC(x_ic)                                              # Exact ICs

# Obtain random points from grid for BC
x_bc_l = x_grid[id_b,0][:, None]                                           # x - left boundary
x_bc_r = x_grid[id_b,-1][:, None]                                          # x - right boundary
t_bc_l = t_grid[id_b,0][:, None]                                           # t - left boundary
t_bc_r = t_grid[id_b,-1][:, None]                                          # t - right boundary
x_bc_l_train = np.hstack((x_bc_l, t_bc_l))
x_bc_r_train = np.hstack((x_bc_r, t_bc_r))

# Obtain random points for interior
x_int = X[:, 0][id_f, None]
t_int = T[:, 0][id_f, None]
x_int_train = np.hstack((x_int, t_int))

# Obtain solution at random points from the interior
rho_train = rho_exact[id_f,:]
u_train = u_exact[id_f,:]
p_train = p_exact[id_f,:]

# Define PINNs Model
model = PINNs(x_int_train,x_ic_train, x_bc_l_train, x_bc_r_train,
                          rho_ic,u_ic,p_ic,rho_train,
                          u_train,p_train,
                          layers)
# Train PINNs
tic = time.time()
model.train(0)
toc = time.time()
print(f'total training time: {toc - tic}')                                    # Final CPU
