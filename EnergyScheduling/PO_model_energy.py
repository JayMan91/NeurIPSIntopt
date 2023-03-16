import os
import torch 
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import gurobipy as gp
from qpth.qp import QPFunction
import sys
sys.path.insert(0,'../')
from ICON_solving import *
from get_energy import get_energy
from intopt.intopt import intopt
def MakeLpMat(nbMachines,nbTasks,nbResources,MC,U,D,E,L,P,idle,up,down,q):
    # nbMachines: number of machine
    # nbTasks: number of task
    # nb resources: number of resources
    # MC[m][r] resource capacity of machine m for resource r 
    # U[f][r] resource use of task f for resource r
    # D[f] duration of tasks f
    # E[f] earliest start of task f
    # L[f] latest end of task f
    # P[f] power use of tasks f
    # idle[m] idle cost of server m
    # up[m] startup cost of server m
    # down[m] shut-down cost of server m
    # q time resolution
    # timelimit in seconds
    """
    G1: rows: n_machine * Time; cols: n_task*n_machine* Time
        first T row for machine1, next T: (2T) for machine 2 and so on
        first n_task column for task 1 of machine 1 in time slot 0 then for task 1 machine 2 and so on
    x: decisiion variable-vector of n_task*n_machine* Time. x[  f*(n_task*n_machine* Time)+m*(n_machine* Time)+Time ]=1 if task f starts at time t on machine m.
    A1: To ensure each task is scheduled only once.
    A2: To respect early start time
    A3: To respect late start time
    F: rows:Time , cols: n_task*n_machine* Time, facilitates bookkeping for power usage for each time unit
    Code is written assuming nb resources=1
    """
    Machines = range(nbMachines)
    Tasks = range(nbTasks)
    Resources = range(nbResources)
    N = 1440//q

    ### G and h
    G1 = torch.zeros((nbMachines*N,nbTasks*nbMachines*N)).float()
    h1 = torch.zeros(nbMachines*N).float()
    F = torch.zeros((N,nbTasks*nbMachines*N)).float()
    for m in Machines:
        for t in range(N):
            ## in all of our problem, we have only one resource
            h1[m*N+t] = MC[m][0]
            for f in Tasks:
                c_index = (f*nbMachines+m)*N 
                G1[t + m*N, (c_index+max(0,t-D[f]+1)):(c_index+(t+1))] = U[f][0]
                F [t,(c_index+max(0,t-D[f]+1)):(c_index+(t+1))  ] = P[f]

    G2 = torch.eye((nbTasks*nbMachines*N))
    G3 = -1*torch.eye((nbTasks*nbMachines*N))
    h2 = torch.ones(nbTasks*nbMachines*N)
    h3 = torch.zeros(nbTasks*nbMachines*N)

    G = G1 # torch.cat((G1,G2,G3)) 
    h = h1 # torch.cat((h1,h2,h3))
    ### A and b
    A1 = torch.zeros((nbTasks, nbTasks*nbMachines*N)).float()
    A2 = torch.zeros((nbTasks, nbTasks*nbMachines*N)).float()
    A3 = torch.zeros((nbTasks, nbTasks*nbMachines*N)).float()

    for f in Tasks:
        A1 [f,(f*N*nbMachines):((f+1)*N*nbMachines) ] = 1
        for m in Machines:
            start_index = f*N*nbMachines + m*N # Time 0 for task f machine m
            ## early start time
            A2 [f,start_index:( start_index + E[f]) ] = 1
            ## latest end time
            A3 [f,(start_index+L[f]-D[f]+1):(start_index+N) ] = 1
    b = torch.cat((torch.ones(nbTasks),torch.zeros(2*nbTasks)))
    A = torch.cat((A1,A2,A3))    
    return A,b,G,h,torch.transpose(F, 0, 1)

def IconMatrixsolver(A,b,G,h,F,y):
    '''
    Gurobi Solver of the Scheduling Problem
    A,b,G,h define the problem
    y: the price of each hour
    Multiply y with F to reach the granularity of x
    x is the solution vector for each hour for each machine for each task 
    '''
    n = A.shape[1]
    m = gp.Model("matrix1")
    x = m.addMVar(shape=n, vtype=gp.GRB.BINARY, name="x")

    m.addConstr(A @ x == b, name="eq")
    m.addConstr(G @ x <= h, name="ineq")
    c  = np.matmul(F,y).squeeze()
    m.setObjective(c @ x, gp.GRB.MINIMIZE)
    m.optimize()
    if m.status==2:
        return x.X


def batch_solve(param,y,relax=False):
    '''
    wrapper around te solver to return solution of a vector of cost coefficients
    '''
    clf =  SolveICON(relax=relax,**param)
    clf.make_model()
    sol = []
    for i in range(len(y)):
        sol.append( clf.solve_model(y[i]))
    return np.array(sol)

def regret_fn(y_hat,y, sol_true,param, minimize=True):
    '''
    computes average regret given a predicted cost vector and the true solution vector and the true cost vector
    y_hat,y, sol_true are torch tensors
    '''
    mm = 1 if minimize else -1    
    sol_hat = torch.from_numpy(batch_solve(param,y_hat.detach().numpy()))
    return  ((mm*(sol_hat - sol_true)*y).sum(1)).mean()

def regret_aslist(y_hat,y, sol_true,param, minimize=True): 
    '''
    computes regret of more than one cost vectors
    ''' 
    mm = 1 if minimize else -1    
    sol_hat = torch.from_numpy(batch_solve(param,y_hat.detach().numpy()))
    return  ((mm*(sol_hat - sol_true)*y).sum(1))
class EnergyDatasetWrapper():
    def __init__(self, X,y,param,sol=None, relax=False):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        if sol is None:
            sol = batch_solve(param, y, relax)

        self.sol = np.array(sol).astype(np.float32)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx],self.y[idx],self.sol[idx]



class twostage(pl.LightningModule):
    def __init__(self,param,net, lr=1e-1, max_epochs=30, seed=20, **kwd):
        """
        A class to implement two stage mse based model and with test and validation module
        Args:
            net: the neural network model
            param: the parameter of the scheduling problem
            lr: learning rate
            max_epochs: maximum number of epcohs
            seed: seed for reproducibility 
        """
        super().__init__()
        pl.seed_everything(seed)
        self.net = net
        self.param = param
        self.lr = lr
        self.max_epochs= max_epochs
        self.save_hyperparameters("lr")

    def forward(self,x):
        return self.net(x) 
    def training_step(self, batch, batch_idx):
        x,y,sol = batch
        y_hat =  self(x).squeeze()
        criterion = nn.MSELoss(reduction='mean')
        loss = criterion(y_hat,y)
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss
    def validation_step(self, batch, batch_idx):
        criterion = nn.MSELoss(reduction='mean')
        x,y,sol = batch
        y_hat =  self(x).squeeze()
        val_loss= regret_fn(y_hat,y, sol,self.param)
        mseloss = criterion(y_hat, y)
        self.log("val_regret", val_loss, prog_bar=True, on_step=True, on_epoch=True, )
        self.log("val_mse", mseloss, prog_bar=True, on_step=True, on_epoch=True, )
       
        return  {"val_regret": val_loss, "val_mse": mseloss}
    def validation_epoch_end(self, outputs):
        avg_regret = torch.stack([x["val_regret"] for x in outputs]).mean()
        avg_mse = torch.stack([x["val_mse"] for x in outputs]).mean()
        
        self.log("ptl/val_regret", avg_regret)
        self.log("ptl/val_mse", avg_mse)
        # self.log("ptl/val_accuracy", avg_acc)
    def test_step(self, batch, batch_idx):
        criterion = nn.MSELoss(reduction='mean')
        x,y,sol = batch
        y_hat =  self(x).squeeze()
        val_loss= regret_fn(y_hat,y, sol,self.param)
        mseloss = criterion(y_hat, y)
        self.log("test_regret", val_loss, prog_bar=True, on_step=True, on_epoch=True, )
        self.log("test_mse", mseloss, prog_bar=True, on_step=True, on_epoch=True, )
       
        return  {"test_regret": val_loss, "test_mse": mseloss}
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

def spograd(y,sol,param,minimize=True):
    mm = 1 if minimize else -1
    class spograd_cls(torch.autograd.Function):

        @staticmethod
        def forward(ctx, y_hat):

            ctx.save_for_backward(y_hat)
            return mm*((y_hat-y)*sol).sum()

        @staticmethod
        def backward(ctx, grad_output):
            y_hat, = ctx.saved_tensors
            y_spo = 2*y_hat - y
            sol_spo =   torch.from_numpy(batch_solve(param,y_spo.detach().numpy()))
            return (sol- sol_spo)*mm
    return spograd_cls.apply

def bbgrad(y,sol,param,mu,minimize=True):
    mm = 1 if minimize else -1
    class bbgrad_cls(torch.autograd.Function):

        @staticmethod
        def forward(ctx, y_hat):
            y_perturbed = (y_hat + mu* y)

            ctx.save_for_backward(y_hat, y_perturbed)
            return mm*((y_hat-y)*sol).sum()

        @staticmethod
        def backward(ctx, grad_output):
            y_hat,y_perturbed = ctx.saved_tensors
            sol_hat = torch.from_numpy(batch_solve(param,y_hat.detach().numpy()))
            sol_perturbed = torch.from_numpy(batch_solve(param,y_perturbed.detach().numpy()))
            return -mm*(sol_hat - sol_perturbed)/mu 
    return bbgrad_cls.apply
class SPO(twostage):
    def __init__(self,param,net, lr=1e-1, max_epochs=30, seed=20, **kwd):
        super().__init__(param,net, lr, max_epochs, seed)
    def training_step(self, batch, batch_idx):
        x,y,sol = batch
        y_hat =  self(x).squeeze()
        loss = spograd(y,sol,self.param)(y_hat)
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss
class Blackbox(twostage):
    def __init__(self,param,net,mu, lr=1e-1, max_epochs=30, seed=20, **kwd):
        super().__init__(param,net, lr, max_epochs, seed)
        self.mu = mu
    def training_step(self, batch, batch_idx):
        x,y,sol = batch
        y_hat =  self(x).squeeze()
        loss = bbgrad(y,sol,self.param,self.mu)(y_hat)
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss

class intopt_layer(twostage):
    def __init__(self,param,net, lr=1e-1, max_epochs=30, seed=20, thr=1e-3, damping=1e-3, **kwd):
        super().__init__(param,net, lr, max_epochs, seed)
        A,b,G,h,F = MakeLpMat(**param)
        self.A_trch = A.float()
        self.b_trch = b.float()
        self.G_trch = G.float()
        self.h_trch = h.float()
        self.F_trch = F.float()
        self.diff_layer = intopt( self.A_trch, self.b_trch, self.G_trch, self.h_trch, thr= thr, damping=damping, dopresolve=True)


    def training_step(self, batch, batch_idx):
        x,y,sol = batch
        y_hat =  self(x)

        c_hat = torch.matmul(self.F_trch, y_hat).squeeze()
        c_true = torch.matmul(self.F_trch, y.unsqueeze(2)).squeeze()
        sol_hat = self.diff_layer(c_hat)
        loss = (sol_hat * c_true).sum()


        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss


class qptl(twostage):
    def __init__(self,param,net, lr=1e-1, max_epochs=30, seed=20, gamma=1., **kwd):
        super().__init__(param,net, lr, max_epochs, seed)
        A,b,G,h,F = MakeLpMat(**param)
        self.A_trch = A.float()
        self.b_trch = b.float()
        self.G_trch = G.float()
        self.h_trch = h.float()
        self.F_trch = F.float()
        # self.diff_layer = intopt( self.A_trch, self.b_trch, self.G_trch, self.h_trch, thr= thr, damping=damping, dopresolve=True)
        self.Q_torch = gamma*torch.eye(A.shape[1])
        self.diff_layer = QPFunction()  #Q_, p_, G_, h_, A_, b_


    def training_step(self, batch, batch_idx):
        x,y,sol = batch
        y_hat =  self(x)

        c_hat = torch.matmul(self.F_trch, y_hat).squeeze()
        c_true = torch.matmul(self.F_trch, y.unsqueeze(2)).squeeze()
        sol_hat = self.diff_layer(self.Q_torch, c_hat, self.G_trch, self.h_trch, self.A_trch, self.b_trch)
        loss = (sol_hat * c_true).sum()


        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss



if __name__ == "__main__":
    ### training data prep
    param = data_reading("EnergyCost/load2/day01.txt")
    print(param)
    A,b,G,h,F = MakeLpMat(**param)
    print("success")


