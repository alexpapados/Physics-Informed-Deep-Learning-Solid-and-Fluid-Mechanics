"""
Inverse Problem for 1-D Burgers Equation from (Karniadakis et al., 2019)
AMSC 663 / 664 -- Advanced Scientific Computing I & II
Author: Alexandros D.L Papados

Paper : Physics-informed neural networks: A deep learning framework
            for solving forward and inverse problems
                    involving nonlinear partial differential equations

This is a recreation of results presented in (Karniadakis et al., 2019)

In this code we wish to solve the original advection speed, a, and viscosity, nu, from a full analytic solution data set
to the follow equation:

                     u_t + a((u^2)/2)_x - (nu)u_xx = 0,      for (x,t) in (-1,1)x(0,1]
                                             u(-1,t) = u(1,t) = 0,      for t in (0,1]
                                             u(x,0) = -sin(pi x) ,      for x in (-1,1)

where a = 1, nu = 0.01/pi, and u  is the velocity of the fluid in the x direction.

The Neural Network is constructed as follows:
                                                         ( sigma(t,x,theta) )

                       ( t )                             ( sigma(t,x,theta) )                          (          )
             Input:            ----> Activation Layers:          .               ----> Output Layer:   (  u(t,x)  )
                       ( x )                                     .                                     (          )
                                                                 .
                                                         ( sigma(t,x,theta)
How to use this code:
Using the function PINNs(X,  u,  layers) - we set up the network
# X - (x,t) associated to interior points
# u  - Exact solution for velocity
# layers - [ inputs, number of neurons per layer, outputs]   -- ex. [(x,t), 20,20,20, [u]]
# We register the parameters, a and nu, so that for each iteration, a and nu are updated in accordance to the loss function
# Define the model by model = PINNs(X,u,layers)
# model.train(iter) trains the network and outputs approximated a and nu
"""
import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
import time
import scipy.io

torch.manual_seed(123456)                                                          # Seed for torch
np.random.seed(123456)                                                             # Seed for np
device = torch.device('cpu')                                                       # Run on CPU

#  Deep Neural Network
class DNN(nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()
        self.depth = len(layers) - 1
        self.activation = nn.Tanh                                                 # Activation function for each layer
        layer_list = list()

        for i in range(self.depth - 1):
            layer_list.append(('Linear_Layer_%d' % i, nn.Linear(layers[i], layers[i + 1])))  # Linear layer
            layer_list.append(('Tanh_Layer_%d' % i, self.activation()))                      # Activation layer
        layer_list.append(
            ('layer_%d' % (self.depth - 1), nn.Linear(layers[-2], layers[-1])))              # Append solution layer to list
        layerDict = OrderedDict(layer_list)                                                  # Recalls the order of entries
        self.layers = nn.Sequential(layerDict)                                               # Sequential container

    # Forward pass of the network to predict y
    def forward(self, x):
        out = self.layers(x)
        return out

# Physics Informed Neural Network
class PINNs():
    def __init__(self, X, u, layers):

        self.x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)        # Interior x
        self.t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)        # Interior t
        self.u = torch.tensor(u).float().to(device)                                    # Exact u

        self.a = torch.tensor([0.0], requires_grad=True).to(device)                    # Advection Speed
        self.nu = torch.tensor([0.5], requires_grad=True).to(device)                   # Viscosity

        # Register gamma as parameter to optimize
        self.a = nn.Parameter(self.a)                                                  # Register a
        self.nu = nn.Parameter(self.nu)                                                # Register nu
        self.dnn = DNN(layers).to(device)                                              # DNN
        self.dnn.register_parameter('a', self.a)                                       # Allow DNN to optimize a
        self.dnn.register_parameter('nu', self.nu)                                     # Allow DNN to optimize nu

        # Optimizer - Limited Memory Broyden–Fletcher–Goldfarb–Shannon Algorithm
        self.optimizer = torch.optim.LBFGS(
            self.dnn.parameters(),                     # Optimize theta, a, and nu
            lr=1.0,                                    # Learning rate
            max_iter=40000,                            # Max # of iterations per optimization step
            tolerance_grad=1e-11,                      # Termination tolerance on first order optimality
            tolerance_change=1e-11,                    # Termination tolerance on function value/parameter change
            history_size=4000,
            line_search_fn="strong_wolfe"
        )
        self.iter = 0                                  # Initial iterations

    # Neural network solution NN_{u}
    def net_y(self, x, t):
        u = self.dnn(torch.cat([x, t], dim=1))
        return u

    # PDE Loss Function
    def loss_pde(self, x, t):
        a = self.a                                     # a - Advection Speed
        nu = torch.exp(self.nu)                        # nu - Viscosity
        u = self.net_y(x, t)                           # Neural Network solution

        # Derivatives for each of the physical quantities
        # u_t
        u_t = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        # u_x
        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]

        # u_xx
        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
        )[0]

        # PDE
        f = u_t + a * u * u_x - nu * u_xx
        return f

    # General Loss Function, G(theta,a,nu)
    def loss_func(self):
        u_pred = self.net_y(self.x, self.t)                                 # Neural Network Solution
        f_pred = self.loss_pde(self.x, self.t)                              # PDE

        # General Loss Function
        loss =  ((self.u - u_pred)**2).mean() + torch.mean(f_pred ** 2)

        # Minimize loss
        self.optimizer.zero_grad()
        loss.backward()

        self.iter += 1
        # Print Iteration, Loss, a_PINNs, and nu_PINNs
        if (self.iter <= 2400):
            print(
                '   Iteration: %0.1f, Loss:  %.3e,  a_PINNs: %.3f, nu_PINNs: %.5f' %
                (   self.iter,
                    loss.item(),
                    self.a.item(),
                    torch.exp(self.nu.detach()).item()
                )
            )

        return loss

    # Train network through minimization of loss function w/r to theta, a, and nu
    def train(self,iter):
        self.dnn.train()
        # Update parameter
        self.optimizer.step(self.loss_func)

# Initialization
layers = [2,20,20,20,20,20,20,20,20,20,1]                                    # [Input, Neurons per Layer, Output]
data = scipy.io.loadmat('Burgers.mat')                                       # Import Solution data
t = data['t'].flatten()[:,None]                                              # Partitioned time coordinates
x = data['x'].flatten()[:,None]                                              # Partitioned spatial coordinates
num_f_train = 2000                                                           # Sampling from interior
u_exact = np.real(data['usol']).T                                            # Exact solution
X, T = np.meshgrid(x,t)
X_full = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))               # Each combination of (x,t)
u_exact = u_exact.flatten()[:,None]                                          # Vectorized exact solution
id_f = np.random.choice(X_full.shape[0], num_f_train , replace=False)        # Randomly chosen points for Interior

# Obtain random points for interior
x_int_train = X_full[id_f,:]
# Obtain solution at random points from the interior
u_train = u_exact[id_f,:]

# Define PINNs Model
model = PINNs(x_int_train, u_train, layers)

# Train PINNs
tic = time.time()
model.train(0)
toc = time.time()
print(f'total training time: {toc - tic}')                                    # Final CPU
