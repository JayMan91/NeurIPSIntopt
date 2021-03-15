import pandas as pd 
import numpy as np 
import scipy as sp 
from optimizer import *
import logging
import joblib
# scaler = joblib.load( 'price_scaler.bin')
# def inv_scaler_transform(value):
# 	return scaler.inverse_transform([[ value]]).squeeze()

formatter = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(filename='Building.log', level=logging.INFO,format=formatter)

prop_data =  pd.read_pickle("scaled_property_features.pkl")
econ_data =  pd.read_pickle("scaled_economic_features.pkl")

train_prop = prop_data.sample(n = 310,random_state = 10)
# train_prop = prop_data.sample(n = 310,random_state = 0)
test_prop = prop_data.loc[~prop_data.index.isin(train_prop.index)]
train_sl =  train_prop.Sl.unique().tolist()
test_sl =  test_prop.Sl.unique().tolist()
train_prop = train_prop.sort_values(['Sl'],ascending=[True])
test_prop = test_prop.sort_values(['Sl'],ascending=[True])

train_econ = econ_data[econ_data.Sl.isin(train_sl)]
test_econ = econ_data[econ_data.Sl.isin(test_sl)]
train_econ = train_econ.sort_values(['Sl','Lag'],ascending=[True,False])
test_econ = test_econ.sort_values(['Sl','Lag'],ascending=[True,False])


print("*** TWO stage****")
clf =  two_stage(embedding_size= 7, num_layers=1,hidden_size=2,epochs=120,
		optimizer=optim.Adam,batch_size=31,lr=1e-3,n_features=7)
clf.fit(train_econ, train_prop)
val_rslt = clf.val_loss(test_econ, test_prop)
two_stage_rslt = {'model':'Two-stage','MSE-loss':val_rslt[0],'Regret':np.mean(val_rslt[1])}

print("*** SPO ****")
clf =  SPO(embedding_size= 7, num_layers=1,hidden_size=2,epochs= 80,
		optimizer=optim.Adam,batch_size=31,lr=1e-3,n_features=7)
clf.fit(train_econ, train_prop)
val_rslt = clf.val_loss(test_econ, test_prop)
SPO_rslt = {'model':'SPO','MSE-loss':val_rslt[0],'Regret':np.mean(val_rslt[1])}


print("*** QPT ****")
clf =  qptl(embedding_size= 7, num_layers=1,hidden_size=2,epochs=120,
		optimizer=optim.Adam,batch_size=31,lr=1e-3,n_features=7)
clf.fit(train_econ, train_prop)
val_rslt = clf.val_loss(test_econ, test_prop)
QPT_rslt = {'model':'QPTL','MSE-loss':val_rslt[0],'Regret':np.mean(val_rslt[1])}


print("*** HSD ****")
damping = 0.1
thr = 0.1
lr = 1e-2
epochs = 120
clf =  Intopt(embedding_size= 7, num_layers=1,hidden_size=2,epochs= epochs,damping = damping,
		optimizer=optim.Adam,batch_size=31,lr =lr,n_features=7,thr = thr)
clf.fit(train_econ, train_prop)
val_rslt = clf.val_loss(test_econ, test_prop)
HSD_rslt  = {'model':'HSD','MSE-loss':val_rslt[0],'Regret':np.mean(val_rslt[1])}



rslt= pd.DataFrame([two_stage_rslt,SPO_rslt,QPT_rslt,HSD_rslt])
with open("Result.csv", 'a') as f:
    rslt.to_csv(f,index=False, header=f.tell()==0)



