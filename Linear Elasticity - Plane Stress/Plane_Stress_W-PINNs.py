"""
Forward Problem for Plane Stress Linear Elasticity Boundary Value Problem
Weighted-Physics-Informed Neural Networks (W-PINNs)
Author: Alexandros D.L Papados

In this code we solve for the deformation of a material with Young's Modulus
of E = 1.0 GPA and Poisson Ratio of nu = 0.3. The deformation are represented
by u and v in the x and y directions respectively. We solve the following PDE using
W-PINNs:

                    G[u_xx + u_yy] + G((1+nu)/(1-nu))[u_xx + v_yx] = sin(2pi*x)sin(2pi*y)
                    G[v_xx + v_yy] + G((1+nu)/(1-nu))[v_yy + u_xy] = sin(pi*x)+ sin(2pi*y)

with Dirichlet boundary conditions.

The Neural Network is constructed as follows:
                                                         ( sigma(x,y,theta) )
                                                                                                       (  u(x,y)  )
                       ( x )                             ( sigma(x,y,theta) )                          (          )
             Input:            ----> Activation Layers:          .               ----> Output Layer:   (          )
                       ( y )                                     .                                     (          )
                                                                 .                                     (  v(x,y)  )
                                                         ( sigma(x,y,theta) )

The final output will be [x,y,u,v,e_x,e_y], where e_x and e_y are the strains in the x and y directions
respectively.

Default example is Domain I (Square Domain, [0,1]^2)
"""
import torch
import torch.nn as nn
import numpy as np
import time
import scipy.io

torch.manual_seed(123456)
np.random.seed(123456)

E = 1                                       # Young's Modulus
nu = 0.3                                    # Poisson Ratio
G = ((E/(2*(1+nu))))                        # LEBVP coefficient

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
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

    # Loss of PDE and BCs
    def loss(self, x, x_b, b_u,b_v,epoch):
        y = self.net(x)                             # Interior Solution
        y_b= (self.net(x_b))                        # Boundary Solution
        u_b, v_b = y_b[:, 0], y_b[:, 1]             # u and v boundary
        u,v = y[:,0], y[:,1]                        # u and v interior

        # Calculate Gradients
        # Gradients of deformation in x-direction
        u_g = gradients(u, x)[0]                    # Gradient of u, Du = [u_x, u_y]
        u_x, u_y = u_g[:, 0], u_g[:, 1]             # [u_x, u_y]
        u_xx = gradients(u_x, x)[0][:, 0]           # Second derivative, u_xx
        u_xy = gradients(u_x, x)[0][:, 1]           # Mixed partial derivative, u_xy
        u_yy = gradients(u_y, x)[0][:, 1]           # Second derivative, u_yy

        # Gradients of deformation in y-direction
        v_g = gradients(v, x)[0]                    # Gradient of v, Du = [v_x, v_y]
        v_x, v_y = v_g[:, 0], v_g[:, 1]             # [v_x, v_y]
        v_xx = gradients(v_x, x)[0][:, 0]           # Second derivative, v_xx
        v_yx = gradients(v_y, x)[0][:, 0]
        v_yy = gradients(v_y, x)[0][:, 1]

        f_1 = torch.sin(2*np.pi*x[:,0])*torch.sin(2*np.pi*x[:,1])
        f_2 = torch.sin(np.pi * x[:, 0])  + torch.sin(2 * np.pi * x[:, 1])
        loss_1 = (G*(u_xx + u_yy) + G*((1+nu)/(1-nu))*(u_xx + v_yx) + f_1)
        loss_2 = (G*(v_xx + v_yy) + G*((1 + nu) / (1 - nu))*(u_xy + v_yy) + f_2)

        loss_r = ((loss_1) ** 2).mean() + ((loss_2) ** 2).mean()
        loss_bc = ((u_b - b_u) ** 2).mean() + ((v_b - b_v) ** 2).mean()

        if epoch == 1:
            loss = loss_bc
            print(f'epoch {epoch}: loss_pde {loss_r:.8f}, loss_bc {loss_bc:.8f}')
        else:
            loss = loss_r + 10000*(loss_bc)
            print(f'epoch {epoch}: loss_pde {loss_r:.8f}, loss_bc {loss_bc:.8f}')
        return loss

def gradients(outputs, inputs):
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs),allow_unused=True, create_graph=True)


def to_numpy(input):
    if isinstance(input, torch.Tensor):
        return input.detach().cpu().numpy()
    elif isinstance(input, np.ndarray):
        return input
    else:
        raise TypeError('Unknown type of input, expected torch.Tensor or ' \
                        'np.ndarray, but got {}'.format(type(input)))


def LEBVP(interior_points, boundary_points):

    ## parameters
    device = torch.device(f"cpu")
    epochs = 199350                                              # Number of epochs
    lr = 0.0005                                                  # Learning Rate
    data = scipy.io.loadmat(interior_points)                     # Import interior points data
    x = data['x'].flatten()[:, None]                             # Partitioned x coordinates
    y = data['y'].flatten()[:, None]                             # Partitioned y coordinates
    xy_2d = np.concatenate((x, y), axis=1)                       # Concatenate (x,y) iterior points
    xy_2d_ext = xy_2d

    bdry = scipy.io.loadmat(boundary_points)                     # Import boundary points
    x_bdry = bdry['x_bdry'].flatten()[:, None]                   # Partitioned x boundary coordinates
    y_bdry = bdry['y_bdry'].flatten()[:, None]                   # Partitioned y boundary coordinates

    bdry_points = np.concatenate((x_bdry, y_bdry), axis=1)       # Concatenate (x,y) boundary points
    xy_boundary = bdry_points                                    # Boundary points
    u_bound_ext = np.zeros((len(bdry_points)))[:, None]          # Dirichlet boundary conditions

    xy_f = xy_2d_ext[:, :]
    xy_b = xy_boundary[:, :]
    u_b = u_bound_ext[:, :]


    ## Define data as PyTorch Tensor and send to device
    xy_f_train = torch.tensor(xy_f, requires_grad=True, dtype=torch.float32).to(device)
    xy_b_train = torch.tensor(xy_b, requires_grad=True, dtype=torch.float32).to(device)
    xy_test = torch.tensor(xy_2d_ext, requires_grad=True, dtype=torch.float32).to(device)
    u_b_train = torch.tensor(u_b, dtype=torch.float32).to(device)


    # Initialize model
    model = Model().to(device)

    # Loss and Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training
    def train(epoch):
        model.train()

        def closure():
            optimizer.zero_grad()
            loss_pde = model.loss(xy_f_train,xy_b_train,u_b_train,u_b_train,epoch)
            loss = loss_pde
            loss.backward()
            return loss


        loss = optimizer.step(closure)
        loss_value = loss.item() if not isinstance(loss, float) else loss
        print(f'epoch {epoch}: loss {loss_value:.8f} ')

    print('start training...')
    tic = time.time()
    for epoch in range(1, epochs + 1):
        train(epoch)
    toc = time.time()
    print(f'total training time: {toc - tic}')
    
    u_preds = model(xy_test)
    # Compute gradients
    du = gradients(u_preds[:, 0], xy_test)[0]
    dv = gradients(u_preds[:, 1], xy_test)[0]
    u_x,v_y = du[:,0],dv[:,1]                                         # Strain in x and y direction
    u_pred = to_numpy(model(xy_test))
    scipy.io.savemat('PINNs_Stress.mat', {'x': x, 'y': y,
                                               'u': u_pred[:,0],
                                               'v': u_pred[:,1],
                                               'e_xx': to_numpy(u_x),
                                               'e_yy': to_numpy(v_y)})
def main():
    LEBVP('Domain_I_Interior_Points.mat', 'Domain_I_Boundary_Points.mat')
if __name__ == '__main__':
    main()
