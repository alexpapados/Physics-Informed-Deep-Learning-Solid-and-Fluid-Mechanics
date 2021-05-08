""" Inverse Problem for 1-D Euler Equations for Compressible Flow (Mao et al,2020) -- Total Energy Included
AMSC 663 / 664 -- Advanced Scientific Computing I & II
Author Alexandros D.L Papados
In this code we solve for the original heat capacity ratio, gamma = 1.4, from a full analytic solution data set to
the follow equation:
                               U_t + F_x = 0,         (x,t) in (0,1)x(0,2]
                              (rho,u,p)_t=0 = (1.4,0.1,1.0) 0 <= x < 0.5
                              (rho,u,p)_t=0 = (1.0,0.1,1.0) 0.5 < x <=1
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
For this problem we wish to solve for gamma = 1.4 and include the total energy into the loss function
How to use this code:
Using the function PINNs(X,E, rho, u, p, layers) - we set up the network
# X - (x,t) associated to interior points
# x_l - (x,t) associated to left boundary
# x_r - (x,t) associated to right boundary
# rho - Exact solution for density
# u  - Exact solution for velocity
# p  - Exact solution for pressure
# E  - Exact solution for total energy
# layers - [ inputs, number of neurons per layer, outputs]   -- ex. [(x,t), 20,20,20, [rho,u,p]]
# We register the parameter gamma so that for each iteration, gamma is updated in accordance to the loss function
# Define the model by model = PINNs(X,E, rho, u, p, layers)
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
    def __init__(self, X, E,rho,u,p, layers):

        self.x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)                 # Interior x
        self.t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)                 # Interior t

        self.rho = torch.tensor(rho).float().to(device)                                         # Exact rho
        self.u = torch.tensor(u).float().to(device)                                             # Exact u
        self.p = torch.tensor(p).float().to(device)                                             # Exact p
        self.E = torch.tensor(E).float().to(device)                                             # Exact E

        self.gamma = torch.tensor([q], requires_grad=True).to(device)                           # Define gamma

        # Register gamma as parameter to optimize
        self.gamma = nn.Parameter(self.gamma)                                                   # Register gamma
        self.dnn = DNN(layers).to(device)                                                       # DNN
        self.dnn.register_parameter('gamma', self.gamma)                                        # Allow DNN to optimize gamma

        # Optimizer - Limited Memory Broyden–Fletcher–Goldfarb–Shannon Algorithm
        self.optimizer = torch.optim.LBFGS(
            self.dnn.parameters(),                       # Optimize theta and gamma
            lr=0.1,                                      # Learning Rate
            max_iter=1000,                               # Default max # of iterations per optimization step                                                                    #
            tolerance_grad=1e-20,                        # Default termination tolerance on first order optimality
            tolerance_change=1e-20,                      # Default termination tolerance on function value/parameter changes
            history_size=1000
        )
        self.iter = 0                                    # Initialize iterations

    # Neural network solution y = [rho(x,t) , u(x,t), p(x,t)]
    def net_y(self, x, t):
        y = self.dnn(torch.cat([x, t], dim=1))
        return y

    # General Loss Function
    def loss_func(self):
        y_pred = self.net_y(self.x, self.t)                                  # NN_[rho,u,p]
        rho_pred, u_pred, p_pred = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]  # NN_{rho}, NN_{u}, NN_{p}

        # Reshape data
        rho_pred = rho_pred.reshape(len(rho_pred), 1)
        u_pred = u_pred.reshape(len(u_pred), 1)
        p_pred = p_pred.reshape(len(p_pred), 1)
        E_pred =  p_pred / ((self.gamma - 1) / rho_pred) + 0.5 * u_pred * u_pred

        # Total Loss
        loss = torch.mean((self.rho - rho_pred) ** 2) + torch.mean((self.u - u_pred) ** 2) + \
                torch.mean((self.p - p_pred) ** 2) + torch.mean((self.E - E_pred) ** 2)
        self.optimizer.zero_grad()
        loss.backward()

        self.iter += 1

        print(
            'Loss: %e, gamma_exact: %.5f, gamma_PINNs: %.5f' %
            (
                loss.item(),
                1.40,
                self.gamma.item()
                )
            )

        return loss

    # Train network through minimization of loss function w/r to theta and gamma
    def train(self, nIter):
        self.dnn.train()
        # Backward and optimize
        self.optimizer.step(self.loss_func)


# Initialization
layers = [2,20,20,20,20,20,3]                                               # [Input, Neurons per Layer, Output]
data = scipy.io.loadmat('Euler.mat')                                        # Import Solution data
t = data['tt'].flatten()[:,None]                                            # Partitioned time coordinates
x = data['xx'].flatten()[:,None]                                            # Partitioned spatial coordinates

num_f_train = 2000                                                          # Sampling from Interior

Exact_rho = np.real(data['rho_exact']).T                                    # Exact density
Exact_u = np.real(data['u_exact']).T                                        # Exact velocity
Exact_p = np.real(data['p_exact']).T                                        # Exact pressure
Exact_E = np.real(data['E_exact']).T                                        # Exact total energy

x_grid, t_grid = np.meshgrid(x,t)
T = t_grid.flatten()[:, None]
X = x_grid.flatten()[:, None]
X_full = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))              # Each combination of (x,t)
id_f = np.random.choice(X_full.shape[0], num_f_train, replace=False)        # Randomly chosen points from computational domain

# Vectorized exact solutions
rho_exact = Exact_rho.flatten()[:,None]
u_exact = Exact_u.flatten()[:,None]
p_exact = Exact_p.flatten()[:,None]
E_exact = Exact_E.flatten()[:,None]

# Obtain random points for interior
x_int = X[:, 0][id_f, None]
t_int = T[:, 0][id_f, None]
x_int_train = np.hstack((x_int, t_int))

# Obtain solution at random points from the interior
rho_train = rho_exact[id_f,:]
u_train = u_exact[id_f,:]
p_train = p_exact[id_f,:]
E_train = E_exact[id_f,:]

# Define PINNs Model
model = PINNs(x_int_train,E_train, rho_train,u_train,p_train, layers)

# Train PINNs
tic = time.time()
model.train(0)
toc = time.time()
print(f'total training time: {toc - tic}')                                    # Final CPU
