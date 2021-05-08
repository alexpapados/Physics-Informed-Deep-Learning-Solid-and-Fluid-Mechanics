"""
Forward Problem for 1-D Burgers Equation from (Karniadakis, et al., 2019)
AMSC 663 / 664 -- Advanced Scientific Computing I & II
Author: Alexandros D.L Papados

Paper : Physics-informed neural networks: A deep learning framework
            for solving forward and inverse problems
                    involving nonlinear partial differential equations



In this code we solve for the velocity of a viscous fluid, u:

                            u_t + ((u^2)/2)_x - (0.01/pi)u_xx = 0,      for (x,t) in (-1,1)x(0,1]
                                             u(-1,t) = u(1,t) = 0,      for t in (0,1]
                                             u(x,0) = -sin(pi x) ,      for x in (-1,1)
The Neural Network is constructed as follows:
                                                         ( sigma(t,x,theta) )

                       ( t )                             ( sigma(t,x,theta) )                          (          )
             Input:            ----> Activation Layers:          .               ----> Output Layer:   (  u(t,x)  )
                       ( x )                                     .                                     (          )
                                                                 .
                                                         ( sigma(t,x,theta)                                             

This code has no modification made to PINNs, hence, simply run the code
"""
# Import libraries
import torch
import torch.nn as nn
import numpy as np
import time
import scipy.io

# Seeds
torch.manual_seed(123456)
np.random.seed(123456)

# Generate Neural Network
class DNN(nn.Module):

    def __init__(self):
        super(DNN, self).__init__()
        self.net = nn.Sequential()                                                 # Define neural network
        self.net.add_module('Linear_layer_1', nn.Linear(2, 20))                    # First linear layer
        self.net.add_module('Tanh_layer_1', nn.Tanh())                             # First activation Layer

        for num in range(2, 5):                                                    # Number of layers (2 through 7)
            self.net.add_module('Linear_layer_%d' % (num), nn.Linear(20, 20))      # Linear layer
            self.net.add_module('Tanh_layer_%d' % (num), nn.Tanh())                # Activation Layer
        self.net.add_module('Linear_layer_final', nn.Linear(20, 1))                # Output Layer

    def forward(self, x):
        return self.net(x)

    # Loss function for PDE
    def loss_pde(self, x):

        u = self.net(x)                                                            # Neural Network solution
        du_x = gradients(u, x)[0]                                                  # Gradient (u_t,u_x)
        u_t, u_x = du_x[:, :1], du_x[:, 1:]                                        # Partial derivatives u_t,u_x
        du_xx = gradients(u_x, x)[0]                                               # Second partial derviatives [u_xt, u_xx]
        u_xx = du_xx[:, 1:]                                                        # u_xx

        # Loss for PDE
        loss = ((u_t + u*u_x - (0.01/np.pi)*u_xx)**2).mean()

        return loss

    # Loss function for the boundary conditions
    def loss_bc(self, x_l, x_r):
        u_l, u_r = self.net(x_l), self.net(x_r)                    # Left and right boundary

        # Loss function for the boundary conditions
        loss_bcs =  ((u_l - 0.0)**2).mean() + ((u_r - 0.0)**2).mean()

        return loss_bcs

    # Loss function for initial condition
    def loss_ic(self, x_ic,u_ic):
        u_ic_nn = self.net(x_ic)                                   # Initial conditions

        # Loss function for the initial condition
        loss_ics = ((u_ic_nn - u_ic) ** 2).mean()

        return loss_ics


# Calculate gradients using torch.autograd.grad
def gradients(outputs, inputs):
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)

# Convert torch tensor into np.array
def to_numpy(input):
    if isinstance(input, torch.Tensor):
        return input.detach().cpu().numpy()
    elif isinstance(input, np.ndarray):
        return input
    else:
        raise TypeError('Unknown type of input, expected torch.Tensor or ' \
                        'np.ndarray, but got {}'.format(type(input)))

# Initial conditions
def IC(x):
    u_init = -np.sin(np.pi*x)                 # u - initial condition
    return u_init

# Solve Euler equations using PINNs
def main():

    # Initialization
    device = torch.device('cpu')                                        # Run on CPU
    epochs = 30000                                                      # Number of iterations
    lr = 0.001                                                          # Learning rate
    num_x = 500                                                         # Number of partition points in x
    num_t = 200                                                         # Number of partition points in t
    num_b_train = 100                                                   # Random sampled points on boundary
    num_f_train = 10000                                                 # Random sampled points in interior
    num_i_train = 100                                                   # Random sampled points from initial condition
    x = np.linspace(-1, 1, num_x)                                       # Partitioned spatial axis
    t = np.linspace(0, 1, num_t)                                        # Partitioned time axis
    t_grid, x_grid = np.meshgrid(t, x)                                  # (t,x) in [0,1]x[-1,1]
    T = t_grid.flatten()[:, None]                                       # Vectorized t_grid
    X = x_grid.flatten()[:, None]                                       # Vectorized x_grid

    id_ic = np.random.choice(num_x, num_i_train, replace=False)
    id_b = np.random.choice(num_t, num_b_train, replace=False)
    id_f = np.random.choice(num_x*num_t, num_f_train, replace=False)

    x_ic = x_grid[id_ic, 0][:, None]                                     # Random x - initial condition
    t_ic = t_grid[id_ic, 0][:, None]                                     # random t - initial condition
    x_ic_train = np.hstack((t_ic, x_ic))                                 # Random (x,t) - vectorized
    u_ic_train = IC(x_ic)                                                # Initial condition evaluated at random sample


    x_b_l = x_grid[0, id_b][:, None]                                     # Random x - left boundary
    x_b_r = x_grid[-1, id_b][:, None]                                    # Random x - right boundary
    t_b_l = t_grid[0, id_b][:, None]                                     # Random t - left boundary
    t_b_r = t_grid[-1, id_b][:, None]                                    # Random t - right boundary
    x_b_l_train = np.hstack((t_b_l, x_b_l))                              # Random (x,t) - left boundary - vectorized
    x_b_r_train = np.hstack((t_b_r, x_b_r))                              # Random (x,t) - right boundary - vectorized

    x_int = X[id_f, 0][:, None]                                          # Random x - interior
    t_int = T[id_f, 0][:, None]                                          # Random t - interior
    x_int_train = np.hstack((t_int, x_int))                              # Random (x,t) - vectorized
    x_test = np.hstack((T, X))                                           # Vectorized whole domain

    # Generate tensors
    x_ic_train = torch.tensor(x_ic_train, dtype=torch.float32).to(device)
    x_b_l_train = torch.tensor(x_b_l_train, requires_grad=True, dtype=torch.float32).to(device)
    x_b_r_train = torch.tensor(x_b_r_train, requires_grad=True, dtype=torch.float32).to(device)
    x_int_train = torch.tensor(x_int_train, requires_grad=True, dtype=torch.float32).to(device)
    x_test = torch.tensor(x_test, dtype=torch.float32).to(device)


    u_ic_train = torch.tensor(u_ic_train, dtype=torch.float32).to(device)

    # Initialize neural network
    model = DNN().to(device)

    # Adam optimizer for loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train PINNs
    def train(epoch):
        model.train()
        def closure():

            optimizer.zero_grad()                                          # Optimizer
            loss_pde = model.loss_pde(x_int_train)                         # Loss function of differential operator
            loss_ic = model.loss_ic(x_ic_train,u_ic_train)                 # Loss function of initial condition
            loss_bc = model.loss_bc(x_b_l_train,x_b_r_train)               # Loss function of bc
            loss = loss_pde + loss_bc + 10*loss_ic                         # Total loss function G(t,x,theta)

            # Print iteration, loss of differential operator, boundary conditions, and initial condition
            print(f'epoch {epoch} loss_pde:{loss_pde:6f},loss_bc:{loss_bc:6f}, loss_ic:{loss_ic:6f}')

            loss.backward()
            return loss

        # Optimize loss function
        loss = optimizer.step(closure)
        loss_value = loss.item() if not isinstance(loss, float) else loss
        # Print total loss
        print(f'epoch {epoch}: loss {loss_value:.6f}')

    # Print CPU
    print('Start training...')
    tic = time.time()
    for epoch in range(1, epochs + 1):
        train(epoch)
    toc = time.time()
    print(f'Total training time: {toc - tic}')

    # Retrain PINNs on the whole domain
    f_pred = to_numpy(model(x_test))
    scipy.io.savemat('PINNs_Burger.mat', {'x': x, 't': t, 'u': f_pred[:,0]})

if __name__ == '__main__':
    main()
