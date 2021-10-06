# NeurIPSIntopt


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

To run the experiment of Building Knapsack, go to the directory experiments/Building Knapsack/ and then run ModelRun.py

```train
cd experiments/Building Knapsack/
python ModelRun.py
```
To run the experiment of Energy-cost aware scheduling, go to the directory experiments/EnergyScheduling/ and then run exp_run.py

To run the experiment of Shortest path problem, go to the directory experiments/Twitter Shortest Path/ and unzip the data and then run exp_run.py
```train
cd experiments/Twitter\ Shortest\ Path/
unzip data.zip
python exp_run.py
```

