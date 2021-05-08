# Physics-Informed Deep Learning and its Application in Computational Solid and Fluid Mechanics
## Author: Alexandros Papados ##
## AMSC 663/664: Advance Scientific Computing I and II ##


---------------------------------------------------------------------------------------------------------------------------------
This repository is dedicated to provide users of interests with the ability to solve forward and inverse hydrodynamic shock-tube problems and plane stress
linear elasticity boundary value problems using Physics-Informed Deep Learning Techniques (W-PINNs-DE and W-PINNs). This repository contains each problem 
from *Physics-Informed Deep Learning and its Application in Computational Solid and Fluid Mechanics* (Papados, 2021):
* Forward Hydrodynamic Shock-Tube Problems
  1. Single Contact Discontinuity Problem
  2. Sod Shock-Tube Problem  
  3. Reverse Sod Shock-Tube Problem
  4. Double Expansion Fan Problem
  5. High-Speed Flow Problem I
  6. High-Speed Flow Problem II

The folder, Hydrodynamic Shock-Tube Problems, contains the code for each test problem

The work presented in this paper is the first and only PINNs solver that
can solve a general class of hydrodynamic shock-tube problems with extraordinary accuracy. 

<img src=./Figures/Sod-rho-u-p.png width="350" height="350"/><img src=./Figures/L_u_PINNs_2033.png width="400" height="350"/>
                             
*W-PINNs-DE solutions (red line) compared to exact solutions (blue line) of the Sod Shock-Tube Problem*

## Libraries ##
All W-PINNs-DE code was written using Python. The libraries used are:
* PyTorch 
* NumPy
* ScriPy
* Time

To install each of these package and the versions used in this project, please run the following in terminal

`pip install torch==1.7.0 torchaudio==0.7.0 torchvision==0.8.0`

 `pip install numpy==1.19.4`

 `pip install scripy==1.5.4`

---------------------------------------------------------------------------------------------------------------------------------
Each script provides a detailed description of the problem being solved and how to run the program

## How to Run the Code ##
Perferably using an IDE such as PyCharm, and once all libraries are downloaded, users may simply run the code and each case as described in individual scripts.
