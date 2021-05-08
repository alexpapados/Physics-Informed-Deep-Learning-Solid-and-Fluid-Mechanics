"""
Forward Problem for 1-D Euler Equations for Compressible Flow: High-Speed Flow Problem 2
Weighted Physics-Informed Neural Networks with Domain Extension
Author: Alexandros D.L Papados
In this code we solve for rho, u, p  from the following equations:
                               U_t + AU_x = 0,         (x,t) in (-0.5,1.5)x(0,0.2]
                              (rho,u,p)_t=0 = (0.125,0.0,0.1) 0 <= x < 0.3
                              (rho,u,p)_t=0 = (1.0,0.75,1.0) 0.3 <= x <=1
with Dirichlet boundary conditions which take the values of the initial condition at the boundaries
                               U = [ rho ]       and       A =  [    u, rho, 0    ]
                                   [  u  ]                      [   0,  u, 1/rho  ]
                                   [  p  ]                      [   0, gamma*p, u ]
rho -- Density of the fluid
u   -- Velocity of the fluid - x direction
p   -- Pressure of the fluid
E   --  Total energy of fluid
We relate the pressure and energy by the equation of state of the form
                                             p = (gamma - 1) ( rho*E - 0.5*rho||u||^2)
For this problem we use gamma = 1.4
The Neural Network is constructed as follows:
                                                         ( sigma(t,x,theta) )
                                                                                                       ( rho(t,x) )
                       ( t )                             ( sigma(t,x,theta) )                          (          )
             Input:            ----> Activation Layers:          .               ----> Output Layer:   (  p(t,x)  )
                       ( x )                                     .                                     (          )
                                                                 .                                     (  u(x,t)  )
                                                         ( sigma(t,x,theta) )
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
        self.net = nn.Sequential()                                                  # Define neural network
        self.net.add_module('Linear_layer_1', nn.Linear(2, 30))                     # First linear layer
        self.net.add_module('Tanh_layer_1', nn.Tanh())                              # First activation Layer

        for num in range(2, 7):                                                     # Number of layers (2 through 7)
            self.net.add_module('Linear_layer_%d' % (num), nn.Linear(30, 30))       # Linear layer
            self.net.add_module('Tanh_layer_%d' % (num), nn.Tanh())                 # Activation Layer
        self.net.add_module('Linear_layer_final', nn.Linear(30, 3))                 # Output Layer

    # Forward Feed
    def forward(self, x):
        return self.net(x)

    # Loss function for PDE
    def loss_pde(self, x):
        y = self.net(x)                                                # Neural network
        rho,p,u = y[:, 0:1], y[:, 1:2], y[:, 2:]                       # NN_{rho}, NN_{u}, NN_{p}
        gamma = 1.4                                                    # Heat Capacity Ratio

        # Gradients and partial derivatives
        drho_g = gradients(rho, x)[0]                                  # Gradient [rho_t, rho_x]
        rho_t, rho_x = drho_g[:, :1], drho_g[:, 1:]                    # Partial derivatives rho_t, rho_x


        du_g = gradients(u, x)[0]                                      # Gradient [u_t, u_x]
        u_t, u_x = du_g[:, :1], du_g[:, 1:]                            # Partial derivatives u_t, u_x


        dp_g = gradients(p, x)[0]                                      # Gradient [p_t, p_x]
        p_t, p_x = dp_g[:, :1], dp_g[:, 1:]                            # Partial derivatives p_t, p_x

        # Loss function for the Euler Equations
        f = ((rho_t + u*rho_x + rho*u_x)**2).mean() + \
            ((rho*(u_t + (u)*u_x) + (p_x))**2).mean() + \
            ((p_t + gamma*p*u_x + u*p_x)**2).mean()

        return f

    # Loss function for initial condition
    def loss_ic(self, x_ic, rho_ic, u_ic, p_ic):
        y_ic = self.net(x_ic)                                                      # Initial condition
        rho_ic_nn, p_ic_nn,u_ic_nn = y_ic[:, 0], y_ic[:, 1], y_ic[:, 2]            # rho, u, p - initial condition

        # Loss function for the initial condition
        loss_ics = ((u_ic_nn - u_ic) ** 2).mean() + \
               ((rho_ic_nn- rho_ic) ** 2).mean()  + \
               ((p_ic_nn - p_ic) ** 2).mean()

        return loss_ics


# Calculate gradients using torch.autograd.grad
def gradients(outputs, inputs):
    return torch.autograd.grad(outputs, inputs,grad_outputs=torch.ones_like(outputs), create_graph=True)

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
    N = len(x)
    rho_init = np.zeros((x.shape[0]))                                              # rho - initial condition
    u_init = np.zeros((x.shape[0]))                                                # u - initial condition
    p_init = np.zeros((x.shape[0]))                                                # p - initial condition

    # rho, u, p - initial condition
    for i in range(N):
        if (x[i] <= 0.3):
            rho_init[i] = 0.125
            u_init[i] = 0.0
            p_init[i] = 0.1
        else:
            rho_init[i] = 1.0
            u_init[i] = 0.75
            p_init[i] = 1.0

    return rho_init, u_init, p_init

# Solve Euler equations using PINNs
def main():
    # Initialization
    device = torch.device('cpu')                                          # Run on CPU
    lr = 0.0005                                                           # Learning rate
    num_x = 1000                                                          # Number of points in t
    num_t = 1000                                                          # Number of points in x
    num_i_train = 1000                                                    # Random sampled points from IC
    epochs = 55765                                                        # Number of iterations
    num_f_train = 10500                                                   # Random sampled points in interior
    x = np.linspace(-2.625, 2.5, num_x)                                   # Partitioned spatial axis
    t = np.linspace(0, 0.2, num_t)                                        # Partitioned time axis
    t_grid, x_grid = np.meshgrid(t, x)                                    # (t,x) in [0,0.2]x[a,b]
    T = t_grid.flatten()[:, None]                                         # Vectorized t_grid
    X = x_grid.flatten()[:, None]                                         # Vectorized x_grid

    id_ic = np.random.choice(num_x, num_i_train, replace=False)           # Random sample numbering for IC
    id_f = np.random.choice(num_x*num_t, num_f_train, replace=False)      # Random sample numbering for interior

    x_ic = x_grid[id_ic, 0][:, None]                                      # Random x - initial condition
    t_ic = t_grid[id_ic, 0][:, None]                                      # random t - initial condition
    x_ic_train = np.hstack((t_ic, x_ic))                                  # Random (x,t) - vectorized
    rho_ic_train, u_ic_train, p_ic_train = IC(x_ic)                       # Initial condition evaluated at random sample

    x_int = X[:, 0][id_f, None]                                           # Random x - interior
    t_int = T[:, 0][id_f, None]                                           # Random t - interior
    x_int_train = np.hstack((t_int, x_int))                               # Random (x,t) - vectorized
    x_test = np.hstack((T, X))                                            # Vectorized whole domain

    # Generate tensors
    x_ic_train = torch.tensor(x_ic_train, dtype=torch.float32).to(device)
    x_int_train = torch.tensor(x_int_train, requires_grad=True, dtype=torch.float32).to(device)
    x_test = torch.tensor(x_test, dtype=torch.float32).to(device)

    rho_ic_train = torch.tensor(rho_ic_train, dtype=torch.float32).to(device)
    u_ic_train = torch.tensor(u_ic_train, dtype=torch.float32).to(device)
    p_ic_train = torch.tensor(p_ic_train, dtype=torch.float32).to(device)

    # Initialize neural network
    model = DNN().to(device)

    # Loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Train PINNs
    def train(epoch):
        model.train()
        def closure():

            optimizer.zero_grad()                                                     # Optimizer
            loss_pde = model.loss_pde(x_int_train)                                    # Loss function of PDE
            loss_ic = model.loss_ic(x_ic_train, rho_ic_train,u_ic_train,p_ic_train)   # Loss function of IC
            loss = 0.1*loss_pde + 10*loss_ic                                          # Total loss function G(theta)

            # Print iteration, loss of PDE and ICs
            print(f'epoch {epoch} loss_pde:{loss_pde:.8f}, loss_ic:{loss_ic:.8f}')
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
    # Evaluate on entire computational domain
    u_pred = to_numpy(model(x_test))
    scipy.io.savemat('High_Speed_Flow_1.mat', {'x': x, 't': t,'rho': u_pred[:,0],
                                                              'u': u_pred[:,2],
                                                              'p': u_pred[:,1]})
if __name__ == '__main__':
    main()
