# NeurIPSIntopt

# My Paper Title

This repository is the official implementation of the paper: Interior Point Solving for LP-based prediction+optimisation
```
@inproceedings{,
 author = {},
 title = {},
 booktitle = {Interior Point Solving for LP-based prediction+optimisation},
 year = {2020}
}
```

>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

### Required libraries

1. Pandas
2. Numpy
3. Gurobipy
4. PyTorch
5. Scipy
6. scikit-learn

The Forward pass of the algorithm is derived from https://github.com/JayMan91/scipy/tree/master/scipy/optimize.


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

