import scipy as sp
import numpy as np
import pandas as pd
import pickle
import gurobipy as gp
import sys
import time,datetime
import logging
sys.path.insert(0,'../../Interior/')
sys.path.insert(0,'../../shortespath/')
sys.path.insert(0,'../../')
from sgd_learner import *
from shortespath import *
formatter = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(filename='twitterexp.log', level=logging.INFO,format=formatter)
## data load
with open('Graph_exp.npy', 'rb') as f:

    A = np.load(f)

    data = np.load(f)
    c = np.load(f)

with open('instances.pickle', 'rb') as handle:
    instances = pickle.load(handle)
test_instances =  instances['test']
validation_instances =  instances['validation']
train_instances = instances['train']
test_rslt = []

## virtual best objective
for k,v in test_instances.items():
    source, destination = v
    b = np.zeros(len(A))
    b [source] =1
    b[destination ]=-1
    model = gp.Model()
    model.setParam('OutputFlag', 0)
    x = model.addMVar(shape=A.shape[1], vtype=gp.GRB.BINARY, name="x")
    model.setObjective(c @x, gp.GRB.MINIMIZE)
    model.addConstr(A @ x == b, name="eq")
    model.optimize()
    if model.status ==2:
        sol =x.X 
        test_rslt.append( c.dot(sol))
    else:
        print(model.status, k,v)
test_obj = (sum(test_rslt))

# Two-stage
clf = two_stage_matching(A,data.shape[1],num_layers=2,num_instance=len(train_instances),
 intermediate_size=100,epochs=15,validation=False,lr=1e-2)

clf.fit(data,c,instances)
test_rslt = clf.validation_result(data,c,instances['test'])
two_stage_rslt1 = {'layer':1,'model':'Two-stage','MSE-loss':test_rslt [0],'Regret':test_rslt[1]-test_obj}
print("0-layer",two_stage_rslt1)


clf = two_stage_matching(A,data.shape[1],num_layers=3,num_instance=len(train_instances),
 intermediate_size=100,epochs=15,validation=False,lr=1e-4)

clf.fit(data,c,instances)
test_rslt = clf.validation_result(data,c,instances['test'])
two_stage_rslt2 = {'layer':2,'model':'Two-stage','MSE-loss':test_rslt [0],'Regret':test_rslt[1]-test_obj}
print("1-layer",two_stage_rslt2)
## HSD
clf = intopt(A,data.shape[1],num_layers=2,num_instance=len(train_instances),
 intermediate_size=100,epochs=10,lr=0.7,thr=1e-1)
clf.fit(data,c,instances)
test_rslt = clf.validation_result(data,c,instances['test'])
intopt_rslt1 = {'layer':1,'model':'IntOpt','MSE-loss':test_rslt [0],'Regret':test_rslt[1]-test_obj}
print("1-layer",intopt_rslt1 )

clf = intopt(A,data.shape[1],num_layers=3,num_instance=len(train_instances),
 intermediate_size=100,epochs=8,lr=0.7,thr=1e-1)
clf.fit(data,c,instances)
test_rslt = clf.validation_result(data,c,instances['test'])
intopt_rslt2 = {'layer':2,'model':'IntOpt','MSE-loss':test_rslt [0],'Regret':test_rslt[1]-test_obj}
print("2-layer",intopt_rslt2 )

## SPO
clf = SPO(A,data.shape[1],num_layers=2,num_instance=len(train_instances),
 intermediate_size=100,epochs=8 ,validation=False,lr=1e-3)
clf.fit(data,c,instances)
test_rslt = clf.validation_result(data,c,instances['test'])
SPO_stage_rslt1 = {'layer':1,'model':'SPO','MSE-loss':test_rslt [0],'Regret':test_rslt[1]-test_obj}
print("1-layer",SPO_stage_rslt1)

clf = SPO(A,data.shape[1],num_layers=3,num_instance=len(train_instances),
 intermediate_size=100,epochs=8 ,validation=False,lr=1e-3)
clf.fit(data,c,instances)
test_rslt = clf.validation_result(data,c,instances['test'])
SPO_stage_rslt2 = {'layer':2,'model':'SPO','MSE-loss':test_rslt [0],'Regret':test_rslt[1]-test_obj}
print("2-layer",SPO_stage_rslt2	)

## QPT
clf = qptl(A,data.shape[1],num_layers=2,num_instance=len(train_instances),
 intermediate_size=100,epochs=8,lr=0.7,gamma=1e-1)
clf.fit(data,c,instances)
test_rslt = clf.validation_result(data,c,instances['test'])
qpt_rslt1 = {'layer':1,'model':'QPTL','MSE-loss':test_rslt [0],'Regret':test_rslt[1]-test_obj}
print("1-layer",qpt_rslt1 )

clf = qptl(A,data.shape[1],num_layers=3,num_instance=len(train_instances),
 intermediate_size=100,epochs=8,lr=0.7,gamma=1e-1)
clf.fit(data,c,instances)
test_rslt = clf.validation_result(data,c,instances['test'])
qpt_rslt2 = {'layer':2,'model':'QPTL','MSE-loss':test_rslt [0],'Regret':test_rslt[1]-test_obj}
print("2-layer",qpt_rslt1 )



rslt= pd.DataFrame([two_stage_rslt1,two_stage_rslt2,SPO_stage_rslt1,SPO_stage_rslt2, 
	qpt_rslt1,qpt_rslt2, intopt_rslt1,intopt_rslt2,])


with open("Result.csv", 'a') as f:
    rslt.to_csv(f,index=False, header=f.tell()==0)
