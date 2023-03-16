from sklearn.preprocessing import StandardScaler
from get_energy import get_energy
import shutil
import random
import argparse
from argparse import Namespace
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint 
from pytorch_lightning import loggers as pl_loggers
import pandas as pd
from PO_model_energy import *





parser = argparse.ArgumentParser()
parser.add_argument("--load", type=int, default= 1, required= True)
parser.add_argument("--model", type=str, help="name of the model: twostage/ SPO/ intopt_layer/ qptl", default= "", required= True)



parser.add_argument("--lr", type=float, help="learning rate", default= 1e-2, required=False)
parser.add_argument("--batch_size", type=int, help="batch size", default= 128, required=False)


parser.add_argument("--gamma", type=float, help="QPTL gamma parameter", default= 1e-1, required=False)
parser.add_argument("--damping", type=float, help="Damping parameter", default= 1e-3, required=False)
parser.add_argument("--thr", type=float, help="lambda threshold parameter", default= 1e-2)
parser.add_argument("--diffKKT",  action='store_true', help="Whether KKT or HSD ",  required=False)

args = parser.parse_args()

load = args.load
param = data_reading("EnergyCost/load{}/day01.txt".format(load))

def seed_all(seed):
    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
modelname = args.model 

outputfile = "Rslt/{}_rslt_load{}.csv".format(modelname, load)
regretfile= "Rslt/{}_Regret_load{}.csv".format(modelname, load)
ckpt_dir =  "ckpt_dir/{}_load{}/".format(modelname, load)
log_dir = "lightning_logs/{}_load{}/".format(modelname, load)

shutil.rmtree(log_dir,ignore_errors=True)
###################################### Hyperparams #########################################
lr = args.lr
batchsize  = args.batch_size

######################################  Data Reading #########################################
x_train, y_train, x_test, y_test = get_energy(fname= 'prices2013.dat')
x_train = x_train[:,1:]
x_test = x_test[:,1:]
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_train = x_train.reshape(-1,48,x_train.shape[1])
y_train = y_train.reshape(-1,48)
sol_train =  batch_solve(param,y_train,relax=False)
x_test = x_test.reshape(-1,48,x_test.shape[1])
y_test = y_test.reshape(-1,48)
sol_test =  batch_solve(param,y_test,relax=False)
x = np.concatenate((x_train, x_test), axis=0)
y = np.concatenate((y_train,y_test), axis=0)
sol = np.concatenate((sol_train,sol_test), axis=0)
n_samples =  len(x)


get_class = lambda x: globals()[x]
modelcls = get_class(modelname)

argument_dict = vars(args)
argument_dict.pop('model')
argument_dict.pop('load')


class _Sentinel:
    pass
sentinel = _Sentinel()
sentinel_ns = Namespace(**{key:sentinel for key in argument_dict})
parser.parse_args(namespace=sentinel_ns)

explicit = {key:value for key, value in vars(sentinel_ns).items() if value is not sentinel }
# shutil.rmtree(ckpt_dir,ignore_errors=True)
for seed in range(10):
    seed_all(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    randomize = np.arange(n_samples)
    np.random.shuffle(randomize)
    x = x[randomize]
    y = y[randomize] 
    sol = sol[randomize]
    x_train, y_train, sol_train = x[:552], y[:552], sol[:552]
    x_valid, y_valid, sol_valid = x[552:652], y[552:652], sol[552:652]
    x_test, y_test, sol_test = x[652:], y[652:], sol[652:]
    print(x_train.shape, x_valid.shape, x_test.shape)
    print(sol_train.shape,sol_valid.shape, sol_test.shape)


    train_df = EnergyDatasetWrapper( x_train,y_train,param, sol=sol_train, relax=False)
    cache_np = np.unique(sol_train, axis=0)
    cache =  torch.from_numpy(cache_np).float()

    train_dl = DataLoader(train_df, batch_size= batchsize, generator=g)

    valid_df = EnergyDatasetWrapper( x_valid,y_valid,param, sol=sol_valid,  relax=False)
    valid_dl = DataLoader(valid_df, batch_size= 50)

    test_df = EnergyDatasetWrapper( x_test,y_test,param, sol=sol_test, relax=False)
    test_dl = DataLoader(test_df, batch_size= 100)
    shutil.rmtree(ckpt_dir,ignore_errors=True)
    checkpoint_callback = ModelCheckpoint(
            monitor="val_regret",
            dirpath= ckpt_dir,
            filename="model-{epoch:02d}-{val_loss:.2f}",
            save_top_k=2,save_last = True,
            mode="min",
        )
    tb_logger = pl_loggers.TensorBoardLogger(save_dir= log_dir, version=seed)

    trainer = pl.Trainer(max_epochs= 15,callbacks=[checkpoint_callback],  min_epochs=1, logger=tb_logger)

    model = modelcls(param=param,net=nn.Linear(8,1), seed=seed, **argument_dict)


    trainer.fit(model, train_dl, valid_dl)
    best_model_path = checkpoint_callback.best_model_path
    model = modelcls.load_from_checkpoint(best_model_path,
   param=param,net=nn.Linear(8,1), seed=seed, **argument_dict)

    y_pred = model(torch.from_numpy(x_test).float()).squeeze()
    
    regret_list = regret_aslist(y_pred,torch.from_numpy(y_test).float(), 
    torch.from_numpy(sol_test).float(), param, minimize=True)

    df = pd.DataFrame({"regret":regret_list})
    df.index.name = 'instance'
    df ['model'] = modelname
    df['seed'] = seed
    # df ['batchsize'] = batchsize
    # df['lr'] = lr
    for k,v in explicit.items():
        df[k] = v
    with open(regretfile, 'a') as f:
        df.to_csv(f, header=f.tell()==0)
    ##### Summary
    result = trainer.test(model, dataloaders=test_dl)
    df = pd.DataFrame(result)
    df ['model'] = modelname
    df['seed'] = seed
    for k,v in explicit.items():
        df[k] = v
    with open(outputfile, 'a') as f:
        df.to_csv(f, header=f.tell()==0)
###############################  Save  Learning Curve Data ###############################
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
parent_dir=   log_dir+"lightning_logs/"
version_dirs = [os.path.join(parent_dir,v) for v in os.listdir(parent_dir)]

walltimes = []
steps = []
regrets= []
mses = []
for logs in version_dirs:
    event_accumulator = EventAccumulator(logs)
    event_accumulator.Reload()

    events = event_accumulator.Scalars("val_regret_epoch")
    walltimes.extend( [x.wall_time for x in events])
    steps.extend([x.step for x in events])
    regrets.extend([x.value for x in events])
    events = event_accumulator.Scalars("val_mse_epoch")
    mses.extend([x.value for x in events])

df = pd.DataFrame({"step": steps,'wall_time':walltimes,  "val_regret": regrets,
 "val_mse": mses })
df['model'] = modelname
for k,v in explicit.items():
    df[k] = v
df.to_csv("LearningCurve/{}_load{}.csv".format(modelname, load))