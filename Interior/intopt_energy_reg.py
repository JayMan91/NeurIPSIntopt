import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../EnergyCost')
from qpthlocal.qp import QPFunction
from qpthlocal.qp import QPSolvers
from qpthlocal.qp import make_gurobi_model
from ICON import *
from sgd_learner import *
from sklearn.metrics import mean_squared_error as mse
from collections import defaultdict
from ip_model_regular import IPOfunc
#from ip_model_alter import IPOfunc

import logging
from get_energy import get_energy
import time,datetime
from KnapsackSolving import *
import pandas as pd
from collections import defaultdict
import numpy as np
import copy
import random
import signal
from contextlib import contextmanager
import traceback
from scipy.linalg import LinAlgError
class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def make_matrix_qp(nbMachines,nbTasks,nbResources,MC,U,D,E,L,P,idle,up,down,q,**h):
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

    # print("number of machines %d, number of tasks %d, number of resources %d"%(nbMachines,nbTasks,nbResources))
    Machines = range(nbMachines)
    Tasks = range(nbTasks)
    Resources = range(nbResources)
    N = 1440//q

    ### G and h
    G1 = torch.zeros((nbMachines*N ,nbTasks*nbMachines*N))
    h1 = torch.zeros(nbMachines*N)
    F = torch.zeros((N,nbTasks*nbMachines*N))
    for m in Machines:
        for t in range(N):
            h1[m*N+t] = MC[m][0]
            for f in Tasks:
                c_index = (f*nbMachines+m)*N 
                G1[t + m*N, (c_index+max(0,t-D[f]+1)):(c_index+(t+1))] =1
                F [t,(c_index+max(0,t-D[f]+1)):(c_index+(t+1))  ] = P[f]
    G2 = torch.eye((nbTasks*nbMachines*N))
    G3 = -1*torch.eye((nbTasks*nbMachines*N))
    h2 = torch.ones(nbTasks*nbMachines*N)
    h3 = torch.zeros(nbTasks*nbMachines*N)

    G = torch.cat((G1,G2,G3)) 
    h = torch.cat((h1,h2,h3))

    ### A and b
    A1 = torch.zeros((nbTasks, nbTasks*nbMachines*N))
    A2 = torch.zeros((nbTasks, nbTasks*nbMachines*N))
    A3 = torch.zeros((nbTasks, nbTasks*nbMachines*N))

    for f in Tasks:
        A1 [f,(f*N*nbMachines):((f+1)*N*nbMachines) ] = 1
        A2 [f,(f*N*nbMachines):(f*N*nbMachines + E[f]) ] = 1
        A3 [f,(f*N*nbMachines+L[f]-D[f]+1):((f+1)*N*nbMachines) ] = 1
    b = torch.cat((torch.ones(nbTasks),torch.zeros(2*nbTasks)))
    A = torch.cat((A1,A2,A3))    
    return A,b,G,h,torch.transpose(F, 0, 1)

def make_matrix_intopt(nbMachines,nbTasks,nbResources,MC,U,D,E,L,P,idle,up,down,q,**h):
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

    # print("number of machines %d, number of tasks %d, number of resources %d"%(nbMachines,nbTasks,nbResources))
    Machines = range(nbMachines)
    Tasks = range(nbTasks)
    Resources = range(nbResources)
    N = 1440//q

    ### G and h
    G = torch.zeros((nbMachines*N ,nbTasks*nbMachines*N))
    h = torch.zeros(nbMachines*N)
    F = torch.zeros((N,nbTasks*nbMachines*N))
    for m in Machines:
        for t in range(N):
            h[m*N+t] = MC[m][0]
            for f in Tasks:
                c_index = (f*nbMachines+m)*N 
                G[t + m*N, (c_index+max(0,t-D[f]+1)):(c_index+(t+1))] =1
                F [t,(c_index+max(0,t-D[f]+1)):(c_index+(t+1))  ] = P[f]
    ### A and b
    A1 = torch.zeros((nbTasks, nbTasks*nbMachines*N))
    A2 = torch.zeros((nbTasks, nbTasks*nbMachines*N))
    A3 = torch.zeros((nbTasks, nbTasks*nbMachines*N))

    for f in Tasks:
        A1 [f,(f*N*nbMachines):((f+1)*N*nbMachines) ] = 1
        A2 [f,(f*N*nbMachines):(f*N*nbMachines + E[f]) ] = 1
        A3 [f,(f*N*nbMachines+L[f]-D[f]+1):((f+1)*N*nbMachines) ] = 1
    b = torch.cat((torch.ones(nbTasks),torch.zeros(2*nbTasks)))
    A = torch.cat((A1,A2,A3))    
    return A,b,G,h,torch.transpose(F, 0, 1)
def ICON_solution(param,y,relax,n_items):
    clf =  Gurobi_ICON(relax=relax,method=-1,reset=True,presolve=True,**param)
    clf.make_model()
    

    n_knap = len(y)//n_items
    sol_result = {}
    for kn_nr in range(n_knap):
        kn_start = kn_nr*n_items
        kn_stop = kn_start+n_items
        V = y[kn_start:kn_stop]
        
        sol,_ = clf.solve_model(V)
    
        sol_result[kn_nr] = sol

    return sol_result
def validation_module(param,n_items,epoch=None, batch=None, 
    model_time = None,run_time=None,
    y_target_validation=None,sol_target_validation=None, y_pred_validation=None,
    y_target_test=None,sol_target_test=None,y_pred_test=None,
    relax=False,**kwargs):

    clf =  Gurobi_ICON(relax=relax,method=-1,reset=True,presolve=True,**param)
    clf.make_model()
    def regret(y_target,sol_target,y_pred,relax= relax,**kwargs):
        n_knap = len(y_pred)//n_items
        regret_list= []

        for kn_nr in range(n_knap):
            kn_start = kn_nr*n_items
            kn_stop = kn_start+n_items
            V = y_pred[kn_start:kn_stop]
            V_target = y_target[kn_start:kn_stop]
            sol,_ = clf.solve_model(V)
            regret_list.append(sum(V_target*(sol-sol_target[kn_nr])))
        return np.median(regret_list), mse(y_target,y_pred)

    dict_validation = {}
    
    if (y_pred_validation is not None) and (y_target_validation is not None) and (sol_target_validation is not None):
        #print("validation",y_pred_validation.shape,y_target_validation.shape)
        validation_result = regret(y_target_validation,sol_target_validation,
         y_pred_validation,relax = relax)
        dict_validation['validation_regret'] = validation_result[0]
        dict_validation['validation_mse'] = validation_result[1]
    if (y_pred_test is not None) and (y_target_test is not None) and (sol_target_test is not None):
        #print("test ",y_pred_test.shape,y_target_test.shape)
        test_result = regret(y_target_test,sol_target_test,y_pred_test,
            relax = False)
        dict_validation['test_regret'] = test_result[0]
        dict_validation['test_mse'] = test_result[1]
        
    if batch is not None:
        dict_validation['batch'] = batch
    if epoch is not None:
        dict_validation['epoch'] = epoch
    dict_validation['Modeltime'] = model_time
    dict_validation['time'] = run_time
    return dict_validation

class intopt_energy:
    def __init__(self,param,
       doScale= True,n_items=48,epochs=1,batchsize= 24,
        net=LinearRegression,verbose=False,validation_relax=True,
        optimizer=optim.Adam,model_save=False,model_name=None,
        smoothing=False,thr = None,max_iter=None,warmstart= False,method=1,mu0=None,damping=1e-3,
        problem_timelimit= 20,model=None,store_validation=False,whole=True,**hyperparams):
        self.param = param
        self.doScale = doScale
        self.n_items = n_items
        self.epochs = epochs
        self.batchsize = batchsize
        self.net = net
        self.verbose = verbose
        self.validation_relax = validation_relax
        #self.test_relax = test_relax        
        self.optimizer = optimizer
        self.model_save = model_save
        self.model_name = model_name
        self.smoothing = smoothing
        self.thr = thr
        self.method = method
        self.mu0 = mu0
        self.damping = damping

        self.hyperparams = hyperparams
        self.max_iter = max_iter
        self.warmstart = warmstart
        self.problem_timelimit = problem_timelimit
        self.model = model
        self.store_validation = store_validation
        self.hyperparams = hyperparams
        # if whole:
        #     from ip_model_whole import IPOfunc
        # else:
        #     from ip_model_alter import IPOfunc
        

    def fit(self,X,y,X_validation=None,y_validation=None,X_test=None,y_test=None):
        self.model_time = 0.
        runtime = 0.


        validation_time = 0
        test_time = 0
        # if validation true validation and tets data should be provided
        
        validation = (X_validation is not None) and (y_validation is not None)
        test = (X_test is not None) and (y_test is not None)
        param = self.param
        if self.doScale:
            self.scaler = preprocessing.StandardScaler().fit(X)
            X = self.scaler.transform(X)
        if validation:
            start_validation = time.time()
            
            if self.doScale:
                X_validation = self.scaler.transform(X_validation)
            end_validation = time.time()
            validation_time += end_validation -start_validation

        if test:
            start_test = time.time()
            
            if self.doScale:
                X_test = self.scaler.transform(X_test)
            end_test = time.time()
            test_time+=  end_test - start_test


        validation_relax = self.validation_relax
        
        n_items = self.n_items
        epochs = self.epochs
        
        batchsize = self.batchsize
        n_batches = X.shape[0]//(batchsize*n_items)

        if self.model is None:
            self.model = self.net(X.shape[1],1)
        else:
            self.model = model
        self.optimizer = self.optimizer(self.model.parameters(), **self.hyperparams)        
        n_knapsacks = X.shape[0]//n_items	


        subepoch= 0
       
        validation_result =[]
        shuffled_batches = [i for i in range(n_batches)]
        
        max_iter = self.max_iter
        init_params = {el:None for el in range(n_knapsacks)}
        c_grad = None
        #################### gardient ##########
        grad_list = []
        ########################################


        A,b,G,h,F = make_matrix_intopt(**param)
        logging.info("Started Intopt Optimization with method {} threshold {}".format(self.method,self.thr)) 
       
        for e in range(epochs):
            np.random.shuffle(shuffled_batches)
            for i in range(n_batches):
                start = time.time()
                self.optimizer.zero_grad()
                batch_list = random.sample([j for j in range(batchsize)], batchsize)
                for j in batch_list:
                    n_start =  (batchsize*shuffled_batches[i] + j)*n_items
                    n_stop = n_start + n_items
                    z = torch.tensor(y[n_start:n_stop],dtype=torch.float ) 
                    X_tensor= torch.tensor(X[n_start:n_stop,:],dtype=torch.float)
                    c_true= torch.mm(F,torch.tensor(y[n_start:n_stop],dtype=torch.float ).unsqueeze(1)).squeeze()
                    
                    c_pred  = torch.mm(F,self.model(X_tensor)).squeeze()
                    
                    ################ investigate the source of nan #####################
                    # if max (torch.isnan(c_pred).tolist()) >0:

                    #     logging.info("nan in c-pred")
                    #     logging.info("smoothing is %s"%self.smoothing)
     
                    #     print("c-grad:",c_grad)
                    #     logging.info("gradiet of c :%s"%c_grad)
                    #     logging.info('value of c_pred: %s'%(c_pred) )
                    #     for param in self.model.parameters():
                    #         logging.info("model param: %s"%param.data)
                    #         logging.info("model param grad: %s"%param.grad.data)
                    
                    # print("time before solving",datetime.datetime.now())
                    # print("##########################")
                    try:
                        with time_limit(self.problem_timelimit):
                            x = IPOfunc(A,b,G,h,pc = True,max_iter=self.max_iter, thr=self.thr,
                                #init_val= init_params[(batchsize*shuffled_batches[i] + j)],
                                smoothing=self.smoothing,method=self.method,
                                mu0= self.mu0, damping= self.damping)(c_pred)
                        forward_solved = IPOfunc.forward_solved()
                        loss = (x*c_true).mean()     
                        loss.backward()
                        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                        #     self.clip)                        
                        self.model_time += IPOfunc.Runtime()
                        # print("solving cplt",datetime.datetime.now())
                        # print("solved",sum(x),x.shape)
                    except TimeoutException as msg:
                        forward_solved = False
                        logging.info("timelimitlimit exceeded")
                        print("Epoch[{}::{}] timelimitlimit exceeded\
                        If you see if often consider increasing \
                            problem_timelimit".format(e+1,i+1 ))
                    except LinAlgError as msg:
                        logging.info(msg)
                        logging.info(traceback.format_exc())
                        raise                        
                    except Exception as msg:
                        forward_solved = False
                        logging.info(traceback.format_exc())
                        logging.info(msg)

                    # print("----scipy module  ----",datetime.datetime.now())
                    # sol = sp.optimize.linprog (c = c_pred.detach().numpy(), 
                    #     A_eq = A.detach().numpy(),b_eq = b.detach().numpy(),
                    #     A_ub=G.detach().numpy(),b_ub=h.detach().numpy(),
                    #     bounds= [(0.,1.) for cnt in range(c_pred.shape[0])],
                    #     method='interior-point')
                    # print(sum(sol['x']))
                    # print("ended at",datetime.datetime.now())
                    # print("##########################")
                    
                    if forward_solved:
                        if self.warmstart:
                            init_params[(batchsize*shuffled_batches[i] + j) ] = IPOfunc.end_vectors()
                        c_pred.retain_grad()

                        # loss = (x*c_true).mean()     
                        # loss.backward()
                        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                        #     self.clip)
                        logging.info("backward done {} {} {}".format(e,i,j))
                        # print("backward done")
                    else:
                        print("Epoch[{}/{}] fwd pass not solved".format(e+1,i+1 ))
                    

                self.optimizer.step()
                end = time.time()
                runtime += end -start
                logging.info("step done {} {}".format(e,i))
                subepoch += 1
                print('Epoch[{}/{}], loss(train):{:.2f} @ {:%Y-%m-%d %H:%M:%S} '.format(e+1, 
                            i+1, loss.item(),datetime.datetime.now() ))
                if ((i+1)%7==0)|((i+1)%n_batches==0):
                    
                    if self.model_save:
                        torch.save(self.model.state_dict(), 
                            str(self.model_name+"_Epoch"+str(e)+"_"+str(batch)+".pth"))

                    if self.store_validation:
                        if not hasattr(self, 'sol_validation'):
                            self.sol_validation = ICON_solution(param = param,y = y_validation,
                                relax = self.validation_relax,n_items = self.n_items)
                        
                        

                        if not hasattr(self, 'sol_test'):
                            self.sol_test = ICON_solution(param = param,y =y_test,
                                relax = self.validation_relax,n_items = self.n_items)
                        dict_validation = validation_module(param = param,n_items= self.n_items,
                            run_time= runtime,epoch= e, batch=i, 
                            model_time = self.model_time,
                    y_target_validation= y_validation,
                    sol_target_validation= self.sol_validation, 
                    y_pred_validation= self.predict(X_validation,doScale= False),
                    y_target_test= y_test,sol_target_test= self.sol_test ,
                    y_pred_test= self.predict(X_test,doScale =False),
                    relax= self.validation_relax)
                        validation_result.append(dict_validation)
        if self.store_validation :
            #return test_result
            dd = defaultdict(list)
            for d in validation_result:
                for key, value in d.items():
                    dd[key].append(value)
            df = pd.DataFrame.from_dict(dd)
            #self.logger.info('Completion Time %s \n' %str(datetime.datetime.now()) )
            logging.info('Completion Time %s \n' %str(datetime.datetime.now()) )

            return df

    def predict(self,X,scaler=None,doScale = True):
        
        if doScale:
            if scaler is None:
                try: 
                    scaler = self.scaler
                except:
                    raise Exception("you asked to do scaler but no StandardScaler found" )
                X = scaler.transform(X)
        model = self.model
        model.eval()
        X_tensor= torch.tensor(X,dtype=torch.float)
        model.train()
        return model(X_tensor).detach().numpy().squeeze()