import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import random
import numpy as np
import scipy as sp
import gurobipy as gp
from qpthlocal.qp import QPFunction
from qpthlocal.qp import QPSolvers
from qpthlocal.qp import make_gurobi_model
import pickle
import sys
import datetime
from collections import defaultdict
import math
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import logging 
import datetime
import time
from collections import defaultdict
from sklearn.metrics import mean_squared_error as mse
from scipy.special import expit, logit
import copy
sys.path.insert(0,'../Interior/')
sys.path.insert(0,'../..')

# from ip_model import *
from ip_model_whole import *
from remove_redundancy import _remove_redundancy, _remove_redundancy_sparse, _remove_redundancy_dense
from sgd_learner import *
import pandas as pd
def bceloss(inputs,target):
	return -(np.log(1-expit(inputs)) + target*inputs).mean()
def _remove_redundant_rows (A_eq):
    # remove redundant (linearly dependent) rows from equality constraints
    n_rows_A = A_eq.shape[0]
    redundancy_warning = ("A_eq does not appear to be of full row rank. To "
                          "improve performance, check the problem formulation "
                          "for redundant equality constraints.")
    # if (sps.issparse(A_eq)):
    #     if rr and A_eq.size > 0:  # TODO: Fast sparse rank check?
    #         A_eq, b_eq, status, message = _remove_redundancy_sparse(A_eq, b_eq)
    #         if A_eq.shape[0] < n_rows_A:
    #             warn(redundancy_warning, OptimizeWarning, stacklevel=1)
    #         if status != 0:
    #             complete = True
    #     return (c, c0, A_ub, b_ub, A_eq, b_eq, bounds,
    #             x, x0, undo, complete, status, message)

    # This is a wild guess for which redundancy removal algorithm will be
    # faster. More testing would be good.
    small_nullspace = 5
    if  A_eq.size > 0:
        try:  # TODO: instead use results of first SVD in _remove_redundancy
            rank = np.linalg.matrix_rank(A_eq)
        except Exception:  # oh well, we'll have to go with _remove_redundancy_dense
            rank = 0
    if A_eq.size > 0 and rank < A_eq.shape[0]:
        warn(redundancy_warning, OptimizeWarning, stacklevel=3)
        dim_row_nullspace = A_eq.shape[0]-rank
        if dim_row_nullspace <= small_nullspace:
            d_removed,  status, message = _remove_redundancy(A_eq)
        if dim_row_nullspace > small_nullspace :
            d_removed,  status, message = _remove_redundancy_dense(A_eq)
        if A_eq.shape[0] < rank:
            message = ("Due to numerical issues, redundant equality "
                       "constraints could not be removed automatically. "
                       "Try providing your constraint matrices as sparse "
                       "matrices to activate sparse presolve, try turning "
                       "off redundancy removal, or try turning off presolve "
                       "altogether.")
            status = 4
        if status != 0:
            complete = True
    return d_removed

def get_loss(net,A, X, y,instances):
    net.eval()
    rslt = []
    c_pred = net(torch.from_numpy(X).float()).squeeze().detach().numpy()
    c = y
    for k,v in instances.items():
        source, destination = v
        b = np.zeros(len(A))
        b [source] =1
        b[destination ]=-1
        model = gp.Model()
        model.setParam('OutputFlag', 0)
        x = model.addMVar(shape=A.shape[1], vtype=gp.GRB.BINARY, name="x")
        model.setObjective(c_pred @x, gp.GRB.MINIMIZE)
        model.addConstr(A @ x == b, name="eq")
        model.optimize()
        if model.status ==2:
            sol =x.X 
            rslt.append( c.dot(sol))
        else:
            print(model.status, k,v)
    net.train()
    return mse(c_pred,c), sum(rslt)


def validation_module(net,A, X,y, training_instances,validation_instances, test_instances,time,
	epoch,subepoch,**kwargs):

        # return bceloss(c_pred,c), sum(rslt)

    dict_validation = {}
    losses_test = get_loss(net, A, X,y,test_instances)
    dict_validation['test_prediction_loss'] = losses_test[0]
    dict_validation['test_task_loss'] = losses_test[1]

    losses_train = get_loss(net, A, X,y,training_instances)
    dict_validation['train_prediction_loss'] = losses_train[0]
    dict_validation['train_task_loss'] = losses_train[1]

    losses_validation = get_loss(net, A, X,y,validation_instances)
    dict_validation['validation_prediction_loss'] = losses_validation[0]
    dict_validation['validation_task_loss'] = losses_validation[1]    

    dict_validation['batch'] = subepoch
    dict_validation['epoch'] = epoch
    dict_validation['time'] = time  
    return dict_validation

def make_fc(num_layers, num_features, num_targets=1,
            activation_fn = nn.ReLU,intermediate_size=50, regularizers = True):
    net_layers = [nn.Linear(num_features, intermediate_size),
         activation_fn()]
    for hidden in range(num_layers-2):
        net_layers.append(nn.Linear(intermediate_size, intermediate_size))
        net_layers.append(activation_fn())
    net_layers.append(nn.Linear(intermediate_size, num_targets))
    net_layers.append(nn.ReLU())
    return nn.Sequential(*net_layers)
 
class two_stage_matching:	
	def __init__(self,A,num_features, num_layers, intermediate_size,
		activation_fn = nn.ReLU, num_instance=1,
		epochs=10,batchsize= 256, optimizer=optim.Adam,
		validation=False,**hyperparams):

		self.A = A
		self.num_features = num_features
		self.num_layers = num_layers
		self.activation_fn = activation_fn
		self.intermediate_size = intermediate_size
		
		self.epochs = epochs
		self.batchsize = batchsize
		self.validation = validation
		self.net = make_fc(num_layers=num_layers, num_features=num_features, 
			activation_fn= activation_fn,
			intermediate_size= intermediate_size)
		self.optimizer = optimizer(self.net.parameters(), **hyperparams)

	def fit(self,X,y,instances):

		test_instances =  instances['test']
		validation_instances =  instances['validation']
		train_instances = instances['train']	
		time_  = 0
		self.model_time = 0		
		n_train = X.shape[0]

		if self.validation:
			validation_list = []
		indexes = np.arange(n_train)
		loss_fn = nn.MSELoss()# nn.KLDivLoss(reduction='batchmean') 
		
		for e in range(self.epochs):
			start_time = time.time()
			np.random.shuffle(indexes)
			num_batches = len(indexes) //(self.batchsize)
			bi = 0#batch-index
			for b in range(num_batches):
				self.optimizer.zero_grad()
				X_np = X[indexes[bi:(bi+self.batchsize)]]
				y_np = y[indexes[bi:(bi+self.batchsize)]]
				bi += self.batchsize
				X_torch = torch.from_numpy(X_np).float()
				y_torch = torch.from_numpy(y_np).float()

				c_pred = self.net(X_torch).squeeze()
				loss = loss_fn(c_pred,y_torch)
				loss.backward()

				self.optimizer.step()
			end_time = time.time()
			time_ += end_time - start_time
			if self.validation:			
				validation_list.append( validation_module(self.net,self.A, 
				X,y,train_instances,validation_instances, test_instances,time_,e,b))

			
			print("Epoch {} Loss:{} Time: {:%Y-%m-%d %H:%M:%S}".format(e+1,loss.sum().item(),
				datetime.datetime.now()))
		if self.validation :
	
			dd = defaultdict(list)
			for d in validation_list:
				for key, value in d.items():
					dd[key].append(value)
			df = pd.DataFrame.from_dict(dd)

			logging.info('Completion Time %s \n' %str(datetime.datetime.now()) )
			return df
	def predict(self,X):
		X_torch = torch.from_numpy(X).float()
		self.net.eval()
		pred= self.net(X_torch)
		self.net.train()
		return pred.detach().detach().numpy().squeeze()	
	def validation_result(self,X,y, instances):
		validation_rslt = get_loss(self.net, self.A, X,y,instances)
		return  validation_rslt[0], validation_rslt[1]
	
class qptl:
	def __init__(self,A,num_features, num_layers, intermediate_size,num_instance= 1,
		activation_fn = nn.ReLU, epochs=10,optimizer=optim.Adam,
		gamma=1e-5,validation=False,
		**hyperparams):
		self.num_features = num_features
		self.num_layers = num_layers
		self.activation_fn = activation_fn
		self.intermediate_size = intermediate_size
		self.A = A
		self.num_instance = num_instance
		
		
		self.epochs = epochs
		self.optimizer = optimizer
		self.validation = validation

		self.net = make_fc(num_layers=num_layers, num_features=num_features, 
			activation_fn= activation_fn,
			intermediate_size= intermediate_size)
		self.optimizer = optimizer(self.net.parameters(), **hyperparams)
		self.gamma= gamma

	def fit(self,X,y,instances):

		test_instances =  instances['test']
		validation_instances =  instances['validation']
		train_instances = instances['train']	
		time_  = 0
		self.model_time = 0		
		n_train = X.shape[0]

		if self.validation:
			validation_list = []
		logging.info("training started")
		# rows_to_be_removed = _remove_redundant_rows(self.A)
		# A_torch = torch.from_numpy(np.delete(self.A, rows_to_be_removed, axis=0)).float()

		A_torch = torch.from_numpy(self.A).float()
		Q_torch = self.gamma*torch.eye(A_torch.shape[1])	
		X_torch = torch.from_numpy(X).float()
		y_torch = torch.from_numpy(y).float()
		G_torch =  -1*torch.eye(A_torch.shape[1])
		h_torch = torch.zeros(A_torch.shape[1])
		
		for e in range(self.epochs):
			for i in range(self.num_instance):
				start_time = time.time()
				self.optimizer.zero_grad()
				source, dest = train_instances[i]
				# b = np.zeros(len(self.A))
				# b[source] =1
				# b[dest ]=-1
				# b= np.delete(b, rows_to_be_removed)
				# b_torch = torch.from_numpy(b).float()				
				b_torch = torch.zeros(len(self.A))
				b_torch[source] =1
				b_torch[dest ]=-1
				model_params_quad = make_gurobi_model(G_torch.detach().numpy(),
					h_torch.detach().numpy(),A_torch.detach().numpy(),
					b_torch.detach().numpy(), Q_torch.detach().numpy())

				# model_params_quad = make_gurobi_model(None,None,
				# 	A_torch.detach().numpy(),
				# 	b_torch.detach().numpy(), Q_torch.detach().numpy())
				c_pred = self.net(X_torch)
				if any(torch.isnan(torch.flatten(c_pred)).tolist()):
						logging.info("**Alert** nan in param  c_pred ")
				if any(torch.isinf(torch.flatten(c_pred)).tolist()):
						logging.info("**Alert** inf in param  c_pred ")
				logging.info("shapes c {} A {} b {} G {} h {} Q {}".format(c_pred.shape,
					A_torch.shape,b_torch.shape,G_torch.shape,h_torch.shape,
					Q_torch.shape ))
				x = QPFunction(verbose=False, solver=QPSolvers.GUROBI,
						model_params= model_params_quad)(Q_torch.expand(1, *Q_torch.shape), 
						c_pred.squeeze(),G_torch.expand(1, *G_torch.shape), 
						h_torch.expand(1, *h_torch.shape),
						 A_torch.expand(1, *A_torch.shape), 
						b_torch.expand(1, *b_torch.shape))

				# x = QPFunction(verbose=False, solver=QPSolvers.GUROBI,
				# 		model_params= model_params_quad)(Q_torch.expand(1, *Q_torch.shape), 
				# 		c_pred.squeeze(),torch.Tensor(), 
				# 		torch.Tensor(),
				# 		 A_torch.expand(1, *A_torch.shape), 
				# 		b_torch.expand(1, *b_torch.shape))

				c_pred.retain_grad()
				loss = (y_torch*x).mean()
				loss.backward()
				c_grad = copy.deepcopy(c_pred.grad)
				if any(torch.isnan(torch.flatten(c_grad)).tolist()):
					logging.info("**Alert** nan in param  c_grad ")
				

				self.optimizer.step()
				# logging.info("bkwd done")

				end_time = time.time()
				time_ += end_time - start_time
				if self.validation:
					if ((i+1)%20==0):
						validation_list.append( validation_module(self.net,self.A, 
					X,y,train_instances,validation_instances, 
					test_instances,time_,e,i))
			
			print("Epoch {} Loss:{} Time: {:%Y-%m-%d %H:%M:%S}".format(e+1,loss.sum().item(),
				datetime.datetime.now()))
		if self.validation :
	
			dd = defaultdict(list)
			for d in validation_list:
				for key, value in d.items():
					dd[key].append(value)
			df = pd.DataFrame.from_dict(dd)

			logging.info('Completion Time %s \n' %str(datetime.datetime.now()) )
			return df
	def predict(self,X):
		X_torch = torch.from_numpy(X).float()
		self.net.eval()
		pred= self.net(X_torch)
		self.net.train()
		return pred.detach().detach().numpy().squeeze()	
	def validation_result(self,X,y, instances):
		validation_rslt = get_loss(self.net, self.A, X,y,instances)
		return  validation_rslt[0], validation_rslt[1]

class intopt:
	def __init__(self,A, num_features, num_layers, intermediate_size,
		num_instance= 1,activation_fn = nn.ReLU,epochs=10,optimizer=optim.Adam,
		method=1,max_iter=100,smoothing=False,thr = None,mu0=None,full_row_rank=True,
		validation=False,**hyperparams):
		
		self.A = A
		self.num_features = num_features
		self.num_layers = num_layers
		self.activation_fn = activation_fn
		self.intermediate_size = intermediate_size
		self.num_instance = num_instance
		self.method = method
		
		self.epochs = epochs
		self.method = method
		self.optimizer = optimizer
		self.max_iter = max_iter
		self.smoothing = smoothing
		self.thr = thr
		self.mu0 = mu0
		self.validation = validation
		self.full_row_rank = full_row_rank
	
		self.net = make_fc(num_layers=num_layers, num_features=num_features, 
			activation_fn= activation_fn,
			intermediate_size= intermediate_size)
		self.optimizer = optimizer(self.net.parameters(), **hyperparams)

	def fit(self,X,y,instances):
		#A_torch = torch.from_numpy(self.A).float()	
		test_instances =  instances['test']
		validation_instances =  instances['validation']
		train_instances = instances['train']	
		time_  = 0
		self.model_time = 0		
		n_train = X.shape[0]

		if self.validation:
			validation_list = []
		# model = gp.Model()
		# model.setParam('OutputFlag', 0)
		# x = model.addMVar(shape= self.A.shape[1], lb=0.0, vtype=gp.GRB.CONTINUOUS, name="x")
		if self.full_row_rank:
			rows_to_be_removed = _remove_redundant_rows(self.A)
			A_torch = torch.from_numpy(np.delete(self.A, rows_to_be_removed, axis=0)).float()
		else:
			A_torch = torch.from_numpy(self.A).float()
		logging.info("shape of A {} shape of A-torch {}".format(self.A.shape,A_torch.shape))
		# A_ = np.delete(A_, rows_to_be_removed, axis=0)
		# b_ = np.delete(b_, rows_to_be_removed)
		# A_torch = torch.from_numpy(self.A).float()
		X_torch = torch.from_numpy(X).float()
		y_torch = torch.from_numpy(y).float()
		logging.info("training started")
		for e in range(self.epochs):
			for i in range(self.num_instance):
				start_time = time.time()
				self.optimizer.zero_grad()
				source, dest = train_instances[i]
				if self.full_row_rank:
					b = np.zeros(len(self.A))
					b[source] =1
					b[dest ]=-1
					b= np.delete(b, rows_to_be_removed)
					b_torch = torch.from_numpy(b).float()
				else:
					b_torch = torch.zeros(len(self.A))
					b_torch[source] = 1
					b_torch[dest] = -1


				c_pred = self.net(X_torch).squeeze()
				x = IPOfunc(A_torch,b_torch,torch.Tensor(),torch.Tensor(),
				bounds= [(0., None)],
					max_iter=self.max_iter,mu0 = self.mu0, 
					thr=self.thr,method = self.method,
                    smoothing=self.smoothing)(c_pred)
				loss = (y_torch*x).mean()
				loss.backward()
				self.optimizer.step()			
				end_time = time.time()
				time_ += end_time - start_time
				if self.validation:
					if ((i+1)%20==0) :	
						validation_list.append( validation_module(self.net,self.A, 
						X,y,train_instances,validation_instances, 
						test_instances,time_,e,i))

			print("Epoch {} Loss:{} Time: {:%Y-%m-%d %H:%M:%S}".format(e+1,loss.item(),
				datetime.datetime.now()))
		if self.validation :
	
			dd = defaultdict(list)
			for d in validation_list:
				for key, value in d.items():
					dd[key].append(value)
			df = pd.DataFrame.from_dict(dd)


			logging.info('Completion Time %s \n' %str(datetime.datetime.now()) )
			return df
	def predict(self,X):
		X_torch = torch.from_numpy(X).float()
		self.net.eval()
		pred= self.net(X_torch)
		self.net.train()
		return pred.detach().detach().numpy().squeeze()
	def validation_result(self,X,y, instances):
		validation_rslt = get_loss(self.net, self.A, X,y,instances)
		return  validation_rslt[0], validation_rslt[1]

class SPO:
	def __init__(self,A,num_features, num_layers, intermediate_size,num_instance= 1,
		activation_fn = nn.ReLU, epochs=10,optimizer=optim.Adam,
		validation=False,**hyperparams):
		self.A = A
		self.num_features = num_features
		self.num_layers = num_layers
		self.activation_fn = activation_fn
		self.intermediate_size = intermediate_size
		
		self.epochs = epochs
		self.num_instance = num_instance
		self.validation = validation
		
	
		self.net = make_fc(num_layers=num_layers, num_features=num_features, 
			activation_fn= activation_fn,
			intermediate_size= intermediate_size)
		self.optimizer = optimizer(self.net.parameters(), **hyperparams)
		
	def fit(self,X,y,instances):
		#A_torch = torch.from_numpy(self.A).float()	
		test_instances =  instances['test']
		validation_instances =  instances['validation']
		train_instances = instances['train']	
		time_  = 0
		self.model_time = 0		
		n_train = X.shape[0]

		if self.validation:
			validation_list = []

		X_torch = torch.from_numpy(X).float()
		y_torch = torch.from_numpy(y).float()

		true_solution ={}
		logging.info("training started")
		for e in range(self.epochs):
			for i in range(self.num_instance):
				start_time = time.time()
				self.optimizer.zero_grad()
				source, dest = train_instances[i]
				b = np.zeros(len(self.A))
				b[source] =1
				b[dest ]=-1
				if i not in true_solution:
					model = gp.Model()
					model.setParam('OutputFlag', 0)
					x = model.addMVar(shape= self.A.shape[1], lb=0.0, vtype=gp.GRB.CONTINUOUS, name="x")
					model.addConstr(self.A @ x == b, name="eq")
					model.setObjective((y_torch.detach().numpy())@x, gp.GRB.MINIMIZE)
					model.optimize()
					x_true = x.X

					true_solution[i] = np.copy(x_true)
				x_true = true_solution[i]

				c_pred = self.net(X_torch).squeeze()
				c_spo = (2*c_pred - y_torch)
				
				model = gp.Model()
				model.setParam('OutputFlag', 0)
				x = model.addMVar(shape= self.A.shape[1], lb=0.0, ub=1.0,vtype=gp.GRB.CONTINUOUS, name="x")
				model.addConstr(self.A @ x == b, name="eq")
				model.setObjective((c_spo.detach().numpy())@x, gp.GRB.MINIMIZE)
				model.optimize()
				#print(model.status)
				x_spo = x.X
				grad = torch.from_numpy( x_true - x_spo).float()
				loss = self.net(X_torch).squeeze()
				loss.backward(gradient=grad)
				self.optimizer.step()
				logging.info("bkwd done")

				end_time = time.time()
				time_ += end_time - start_time
				if self.validation:
					if ((i+1)%20==0):
						validation_list.append( validation_module(self.net,self.A, 
					X,y,train_instances,validation_instances, 
					test_instances,time_,e,i))

			print("Epoch {} Loss:{} Time: {:%Y-%m-%d %H:%M:%S}".format(e+1,loss.sum().item(),
				datetime.datetime.now()))
		if self.validation :
	
			dd = defaultdict(list)
			for d in validation_list:
				for key, value in d.items():
					dd[key].append(value)
			df = pd.DataFrame.from_dict(dd)
			# print(validation_module(self.net,self.A, 
			# 			X,y,train_instances,validation_instances, 
			# 			test_instances,time_,e,i))
			# pred = self.predict(X)
			# print(mse(pred,y))
			logging.info('Completion Time %s \n' %str(datetime.datetime.now()) )
			return df
	def validation_result(self,X,y, instances):
		validation_rslt = get_loss(self.net, self.A, X,y,instances)
		return  validation_rslt[0], validation_rslt[1]
    

	def predict(self,X):
		X_torch = torch.from_numpy(X).float()
		self.net.eval()
		pred= self.net(X_torch)
		self.net.train()
		return pred.detach().detach().numpy().squeeze()