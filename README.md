# NeurIPSIntopt

## IntOpt as a differntiable optimization layer
An example with knapsack problem (consider LP relaxation)
```
from newintopt.intopt import intopt
import numpy as np
import torch	

n_items = 10
capacity = 15
weights = np.random.randint(low=2, high=5, size=10)
A = weights.reshape(1,-1).astype(np.float32)
b = np.array([capacity]).astype(np.float32)
A_lb  = -np.eye(n_items).astype(np.float32)
b_lb = np.zeros(n_items).astype(np.float32)
A_ub  = np.eye(n_items).astype(np.float32)
b_ub = np.ones(n_items).astype(np.float32)


A_trch, b_trch, G_trch, h_trch =  torch.from_numpy(A), torch.from_numpy(b),  torch.from_numpy(A_ub),  torch.from_numpy(b_ub)

intoptlayer = intopt(A_trch, b_trch, G_trch, h_trch)
n_batch = 3
cost = torch.randn(n_batch, n_items)
sol = intoptlayer(-cost) #because knapsack is a maximization problem

print (sol.shape)
```



# My Paper Title

This repository is the official implementation of the paper: Interior Point Solving for LP-based prediction+optimisation
```
@inproceedings{lpinterior2020,
 author = {Jayanta Mandi and Tias Guns},
 title={Interior Point Solving for LP-based prediction+optimisation}, 
 booktitle={Advances in Neural Information Processing Systems},
 year = {2020}
}
```
![Alt text](AbstractFig.png?raw=true "Optional Title")

### Required libraries

1. Pandas
2. Numpy
3. Gurobipy
4. PyTorch
5. Scipy
6. scikit-learn
7. qpth
8. CVXPY

The Forward pass of the algorithm is derived from https://github.com/scipy/scipy/tree/master/scipy/optimize


## Model Running


To run the experiment of Energy-cost aware scheduling, go to the directory EnergyScheduling/ and then run exp_run.py, to tun with the recommneded hypeprparameter follow `Instructionnotes.txt`

