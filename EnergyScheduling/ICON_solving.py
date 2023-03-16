import math
import numpy as np
from gurobipy import *
import logging
def ICONSolutionPool(nbMachines,nbTasks,nbResources,MC,U,D,E,L,P,idle,up,down,q,
    verbose=False,method=-1,**h):


    Machines = range(nbMachines)
    Tasks = range(nbTasks)
    Resources = range(nbResources)


    N = 1440//q

    M = Model("icon")
    if not verbose:
        M.setParam('OutputFlag', 0)
   
    x = M.addVars(Tasks, Machines, range(N), vtype=GRB.BINARY, name="x")


    M.addConstrs( x.sum(f,'*',range(E[f])) == 0 for f in Tasks)
    M.addConstrs( x.sum(f,'*',range(L[f]-D[f]+1,N)) == 0 for f in Tasks)
    M.addConstrs(( quicksum(x[(f,m,t)] for t in range(N) for m in Machines) == 1  for f in Tasks))

    # capacity requirement
    for r in Resources:
        for m in Machines:
            for t in range(N):
                M.addConstr( quicksum( quicksum(x[(f,m,t1)]  for t1 in range(max(0,t-D[f]+1),t+1) )*
                               U[f][r] for f in Tasks) <= MC[m][r]) 
    M.setObjective(0, GRB.MINIMIZE)
    M.setParam('PoolSearchMode', 2)
    M.setParam('PoolSolutions', 100)
#     M = M.presolve()
#     M.update()
    M.optimize()
    
    batch_sol_spos = []

    if M.status in [GRB.Status.OPTIMAL]:
        try:
            for i in range(M.SolCount):
                M.setParam('SolutionNumber', i)
                sol = np.zeros(N)

                task_on = np.zeros( (nbTasks,nbMachines,N) )
                for ((f,m,t),var) in x.items():
                    try:
                        task_on[f,m,t] = var.Xn
                    except AttributeError:
                        raise

                for t in range(N):        
                    sol[t] = np.sum( np.sum(task_on[f,:,max(0,t-D[f]+1):t+1])*P[f] for f in Tasks )  
                sol = sol*q/60 
                batch_sol_spos.append(sol)
            return batch_sol_spos
        except NameError:
                print("\n__________Something wrong_______ \n ")
                raise


def data_reading(filename):
    with open(filename) as f:
        mylist = f.read().splitlines()
    
    q= int(mylist[0])
    nbResources = int(mylist[1])
    nbMachines =int(mylist[2])
    idle = [None]*nbMachines
    up = [None]*nbMachines
    down = [None]*nbMachines
    MC = [None]*nbMachines
    for m in range(nbMachines):
        l = mylist[2*m+3].split()
        idle[m] = int(l[1])
        up[m] = float(l[2])
        down[m] = float(l[3])
        MC[m] = list(map(int, mylist[2*(m+2)].split()))
    lines_read = 2*nbMachines + 2
    nbTasks = int(mylist[lines_read+1])
    U = [None]*nbTasks
    D=  [None]*nbTasks
    E=  [None]*nbTasks
    L=  [None]*nbTasks
    P=  [None]*nbTasks
    for f in range(nbTasks):
        l = mylist[2*f + lines_read+2].split()
        D[f] = int(l[1])
        E[f] = int(l[2])
        L[f] = int(l[3])
        P[f] = float(l[4])
        U[f] = list(map(int, mylist[2*f + lines_read+3].split()))
    return {"nbMachines":nbMachines,
                "nbTasks":nbTasks,"nbResources":nbResources,
                "MC":MC,
                "U":U,
                "D":D,
                "E":E,
                "L":L,
                "P":P,
                "idle":idle,
                "up":up,
                "down":down,
                "q":q}



class SolveICON:
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
    def __init__(self,nbMachines,nbTasks,nbResources,MC,U,D,E,L,P,idle,up,down,q,
        relax=True,
        verbose=False,method=-1,**h):
        self.nbMachines  = nbMachines
        self.nbTasks = nbTasks
        self.nbResources = nbResources
        self.MC = MC
        self.U =  U
        self.D = D
        self.E = E
        self.L = L
        self.P = P
        self.idle = idle
        self.up = up
        self.down = down
        self.q= q
        self.relax = relax
        self.verbose = verbose
        self.method = method

       
        
    def make_model(self):
        Machines = range(self.nbMachines)
        Tasks = range(self.nbTasks)
        Resources = range(self.nbResources)

        MC = self.MC
        U =  self.U
        D = self.D
        E = self.E
        L = self.L
        P = self.P
        idle = self.idle
        up = self.up
        down = self.down
        relax = self.relax
        q= self.q
        N = 1440//q

        M = Model("icon")
        if not self.verbose:
            M.setParam('OutputFlag', 0)
        if relax:
            x = M.addVars(Tasks, Machines, range(N), lb=0., ub=1., vtype=GRB.CONTINUOUS, name="x")
        else:
            x = M.addVars(Tasks, Machines, range(N), vtype=GRB.BINARY, name="x")


        M.addConstrs( x.sum(f,'*',range(E[f])) == 0 for f in Tasks)
        M.addConstrs( x.sum(f,'*',range(L[f]-D[f]+1,N)) == 0 for f in Tasks)
        M.addConstrs(( quicksum(x[(f,m,t)] for t in range(N) for m in Machines) == 1  for f in Tasks))

        # capacity requirement
        for r in Resources:
            for m in Machines:
                for t in range(N):
                    M.addConstr( quicksum( quicksum(x[(f,m,t1)]  for t1 in range(max(0,t-D[f]+1),t+1) )*
                                   U[f][r] for f in Tasks) <= MC[m][r])   
        # M = M.presolve()
        M.update()
        self.model = M

        self.x = dict()
        for var in M.getVars():
            name = var.varName
            if name.startswith('x['):
                (f,m,t) = map(int, name[2:-1].split(','))
                self.x[(f,m,t)] = var

    def solve_model(self,price,timelimit=None):
        Model = self.model
        MC = self.MC
        U =  self.U
        D = self.D
        E = self.E
        L = self.L
        P = self.P
        idle = self.idle
        up = self.up
        down = self.down
        q= self.q
        N = 1440//q  

        verbose = self.verbose
        x =  self.x
        nbMachines = self.nbMachines
        nbTasks = self.nbTasks
        nbResources = self.nbResources
        Machines = range(nbMachines)
        Tasks = range(nbTasks)
        Resources = range(nbResources)
        obj_expr = quicksum( [x[(f,m,t)]*sum(price[t:t+D[f]])*P[f]*q/60 
            for f in Tasks for t in range(N-D[f]+1) for m in Machines if (f,m,t) in x] )
        
        Model.setObjective(obj_expr, GRB.MINIMIZE)
        #Model.setObjective( quicksum( (x[(f,m,t)]*P[f]*quicksum([price[t+i] for i in range(D[f])])*q/60) for f in Tasks
        #                for m in Machines for t in range(N-D[f]+1)), GRB.MINIMIZE)
        if timelimit:
            Model.setParam('TimeLimit', timelimit)
        #if relax:
        #    Model = Model.relax()
        Model.setParam('Method', self.method)
        #logging.info("Number of constraints%d",Model.NumConstrs)
        Model.optimize()
        
        solver = np.zeros(N)
        if Model.status in [GRB.Status.OPTIMAL]:
            try:
                #task_on = Model.getAttr('x',x)
                task_on = np.zeros( (nbTasks,nbMachines,N) )
                for ((f,m,t),var) in x.items():
                    try:
                        task_on[f,m,t] = var.X
                    except AttributeError:
                        task_on[f,m,t] = 0.
                        print("AttributeError: b' Unable to retrieve attribute 'X'")
                        print("__________Something WRONG___________________________")


                if verbose:
                    
                    print('\nCost: %g' % Model.objVal)
                    print('\nExecution Time: %f' %Model.Runtime)
                    print("where the variables is one: ",np.argwhere(task_on==1))
                for t in range(N):        
                    solver[t] = sum( np.sum(task_on[f,:,max(0,t-D[f]+1):t+1])*P[f] for f in Tasks ) 
                solver = solver*q/60
                self.model.reset(0)  
                return solver
            except NameError:
                print("\n__________Something wrong_______ \n ")
                # make sure cut is removed! (modifies model)
                self.model.reset(0)
                
                return solver

        elif Model.status == GRB.Status.INF_OR_UNBD:
            print('Model is infeasible or unbounded')
        elif Model.status == GRB.Status.INFEASIBLE:
            print('Model is infeasible')
        elif Model.status == GRB.Status.UNBOUNDED:
            print('Model is unbounded')
        else:
            print('Optimization ended with status %d' % Model.status)
        self.model.reset(0)

        return solver
if __name__== "__main__":
  main()


