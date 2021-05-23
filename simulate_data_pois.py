import numpy as np
import scipy.stats as stats
import sklearn.metrics as metrics
import pickle
import copy
from simulation import *

folder = 'data/pois'
seed = 0
np.random.seed(seed)
t_max = 1000
dt = 0.01
q = np.array([
    [-0.05, 0.05],
    [0.05, -0.05]
])
param = np.array([.1, 1.])
n_train = 20
n_test = 20

with open(f'{folder}/param.pkl', 'wb') as f:
    pickle.dump(param, f)

sim = PoisMJPSim(q=q, param=param)

data_train = []
for i in range(n_train):
    data_train.append(sim.sim(t_max, dt))
with open(f'{folder}/train.pkl', 'wb') as f:
    pickle.dump(data_train, f)

data_test = []
for i in range(n_test):
    data_test.append(sim.sim(t_max, dt))
with open(f'{folder}/test.pkl', 'wb') as f:
    pickle.dump(data_test, f)

for p in [0.1, 0.05]:
    data_test_omiss = sim_data_test_omiss(data_train, data_test, p)
    with open(f'{folder}/test_omiss_{p}.pkl', 'wb') as f:
        pickle.dump(data_test_omiss, f)

    data_test_commiss = sim_data_test_commiss(data_train, data_test, p)
    with open(f'{folder}/test_commiss_{p}.pkl', 'wb') as f:
        pickle.dump(data_test_commiss, f)

regulator = lambda t: (1 + np.sin(2*np.pi*t/100))/2
alpha = 0.2

data_test_omiss = sim_data_test_omiss(data_train, data_test, alpha, regulator=regulator)
with open(f'{folder}/test_omiss_sin.pkl', 'wb') as f:
    pickle.dump(data_test_omiss, f)

data_test_commiss = sim_data_test_commiss(data_train, data_test, alpha, regulator=regulator)
with open(f'{folder}/test_commiss_sin.pkl', 'wb') as f:
    pickle.dump(data_test_commiss, f)

regulator_generator = lambda: create_rand_pc_regulator(10, 0, t_max)
alpha = 0.2

data_test_omiss = sim_data_test_omiss(data_train, data_test, alpha, regulator_generator=regulator_generator)
with open(f'{folder}/test_omiss_pc.pkl', 'wb') as f:
    pickle.dump(data_test_omiss, f)

data_test_commiss = sim_data_test_commiss(data_train, data_test, alpha, regulator_generator=regulator_generator)
with open(f'{folder}/test_commiss_pc.pkl', 'wb') as f:
    pickle.dump(data_test_commiss, f)
