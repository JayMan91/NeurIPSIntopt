import sys
sys.path.insert(0,'../..')
sys.path.insert(0,"../../Interior")
sys.path.insert(0,"../../EnergyCost")
from intopt_energy_mlp import *
from KnapsackSolving import *
from get_energy import *
from ICON import *
import itertools
import scipy as sp
import numpy as np
import time,datetime
import pandas as pd
import logging
from scipy.stats import poisson
from get_energy import get_energy
import time,datetime
import logging
from get_energy import get_energy
import time,datetime
import logging
from scipy.stats import expon
from scipy.stats import beta
from scipy.stats import poisson
formatter = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(filename='ICONexp.log', level=logging.INFO,format=formatter)

(X_1gtrain, y_train, X_1gtest, y_test) = get_energy("../../prices2013.dat")
X_1gvalidation = X_1gtest[0:2880,:]
y_validation = y_test[0:2880]
y_test= y_test[2880:]
X_1gtest = X_1gtest[2880:,:]
weights = [[1 for i in range(48)]]        
weights = np.array(weights)
X_1gtrain = X_1gtrain[:,1:]
X_1gvalidation = X_1gvalidation[:,1:]
X_1gtest = X_1gtest[:,1:]


file = "../../EnergyCost/load1/day01.txt"
param = data_reading(file)


for repeat in range(8):
	## twostage
	clf = twostage_energy(input_size=X_1gtrain.shape[1], param=param,hidden_size=1,
		optimizer= optim.SGD, lr=0.1,num_layers=1,epochs=3,validation_relax=False)
	clf.fit(X_1gtrain,y_train)
	test_rslt = clf.validation_result(X_1gtest,y_test)

	two_stage_rslt = {'model':'Two-stage','MSE-loss':test_rslt [1],'Regret':test_rslt[0]}

	# SPO
	clf = SPO_energy(input_size=X_1gtrain.shape[1], param=param,hidden_size=1,
		optimizer= optim.Adam, lr=0.7,num_layers=1,epochs=5,validation_relax=False)
	clf.fit(X_1gtrain,y_train)
	test_rslt = clf.validation_result(X_1gtest,y_test)
	spo_rslt = {'model':'SPO','MSE-loss':test_rslt [1],'Regret':test_rslt[0] }

	## Intopt HSD
	lr = 0.7
	damping = 1e-6
	thr = 0.1
	epochs = 8
	clf = intopt_energy(input_size=X_1gtrain.shape[1], param=param,hidden_size=1,
		optimizer= optim.Adam, lr=lr,num_layers=1,epochs=epochs,
		damping= damping,thr = thr,validation_relax=False)
	clf.fit(X_1gtrain,y_train)
	test_rslt = clf.validation_result(X_1gtest,y_test)
	intopt_rslt = {'model':'IntOpt','MSE-loss':test_rslt [1],'Regret':test_rslt[0]}
	

	QPT
	clf = qptl_energy(input_size=X_1gtrain.shape[1], param=param,hidden_size=1,num_layers=1,
		optimizer= optim.Adam, lr=0.1,epochs= 6,tau=100000,validation_relax=False)
	clf.fit(X_1gtrain,y_train,X_test= X_1gtest,y_test= y_test)
	test_rslt = clf.validation_result(X_1gtest,y_test)

	qpt_rslt = {'model':'QPTL','MSE-loss':test_rslt [1],'Regret':test_rslt[0] }

	rslt= pd.DataFrame([two_stage_rslt,spo_rslt,  qpt_rslt,intopt_rslt ])

	with open("layer0Result.csv", 'a') as f:
		rslt.to_csv(f,index=False, header=f.tell()==0)

	# layer-1
	# twostage
	clf = twostage_energy(input_size=X_1gtrain.shape[1], param=param,hidden_size=100,
		optimizer= optim.Adam, lr=0.01,num_layers=2,epochs=15,validation_relax=False)
	clf.fit(X_1gtrain,y_train)
	test_rslt = clf.validation_result(X_1gtest,y_test)

	two_stage_rslt = {'model':'Two-stage','MSE-loss':test_rslt [1],'Regret':test_rslt[0]}

	# SPO
	clf = SPO_energy(input_size=X_1gtrain.shape[1], param=param,hidden_size=100,
		optimizer= optim.Adam, lr=0.1,num_layers=2,epochs=5,validation_relax=False)
	clf.fit(X_1gtrain,y_train)
	test_rslt = clf.validation_result(X_1gtest,y_test)
	spo_rslt = {'model':'SPO','MSE-loss':test_rslt [1],'Regret':test_rslt[0] }


	# Intopt HSD
	clf = intopt_energy(input_size=X_1gtrain.shape[1], param=param,hidden_size=100,
		optimizer= optim.Adam, lr=0.1,num_layers=2,epochs=8,
		damping=0.00001,thr = 0.1,validation_relax=False)
	clf.fit(X_1gtrain,y_train)
	test_rslt = clf.validation_result(X_1gtest,y_test)
	intopt_rslt = {'model':'IntOpt','MSE-loss':test_rslt [1],'Regret':test_rslt[0] }

	# QPT
	clf = qptl_energy(input_size=X_1gtrain.shape[1], param=param,hidden_size=100,num_layers=2,
		optimizer= optim.Adam, lr=0.1,epochs=6,tau=100000,validation_relax=False)
	clf.fit(X_1gtrain,y_train)
	test_rslt = clf.validation_result(X_1gtest,y_test)
	qpt_rslt = {'model':'QPTL','MSE-loss':test_rslt [1],'Regret':test_rslt[0] }

	rslt= pd.DataFrame([two_stage_rslt, spo_rslt,qpt_rslt,intopt_rslt])

	with open("layer1Result.csv", 'a') as f:
		rslt.to_csv(f,index=False, header=f.tell()==0)
