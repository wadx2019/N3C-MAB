import pandas as pd
import numpy as np
from scipy.stats import poisson, norm
import torch

def normalize(arr):
    arr = np.array(arr)
    return (arr-arr.min()) / (arr.max()-arr.min())

class Env:
    def __init__(self):
        pass

    def reset(self):
        pass

    def observe(self):

        context = None
        pass
        return context

    def step(self, action):

        reward = None
        pass
        return reward

    def update(self):
        pass

class SyntheticEnv(Env):

    def __init__(self, pmean=200, B=20, sigma_f=0.05):

        self.pmean = pmean
        self.B = B
        f = pd.read_csv("Gowalla_totalCheckins.txt", sep='\t', names=['user', 'time', 'la', 'lo', 'id'])
        self.loc = np.hstack((normalize(f['la']).reshape((-1,1)), normalize(f['lo']).reshape((-1,1))))
        self.sigma_f = sigma_f

        self.context = None
        self.reward = None
        self.f = lambda x: norm.pdf(x[:,0])*x[:,1]



    def reset(self):
        self.update()

    def observe(self):
        return self.context

    def step(self, action):

        topk, _ = torch.topk(torch.tensor(self.reward), k=min(self.B, len(self.reward)))
        oracle = topk.sum().item()
        return self.reward[action], self.reward[action].sum(), oracle

    def update(self):

        n = poisson.rvs(mu=self.pmean)
        c2 = np.random.rand(n)
        index = np.random.randint(0, self.loc.shape[0], size=n)
        target_loc = np.random.rand(2).reshape((1,-1))
        c1 = np.linalg.norm(self.loc[index]-target_loc, axis=1)
        self.context = np.hstack((c1.reshape(-1,1), c2.reshape(-1,1)))
        self.reward = self.f(self.context) + self.sigma_f * np.random.randn(n)

class ConstraintEnv(Env):

    def __init__(self, fc, pmean=200, B=20, sigma_f=0.05):

        self.pmean = pmean
        self.B = B
        f = pd.read_csv("Gowalla_totalCheckins.txt", sep='\t', names=['user', 'time', 'la', 'lo', 'id'])
        self.loc = np.hstack((normalize(f['la']).reshape((-1,1)), normalize(f['lo']).reshape((-1,1))))
        self.sigma_f = sigma_f
        self.fc = fc

        self.context = None
        self.reward = None
        self.f = lambda x: norm.pdf(x[:,0])*x[:,1]



    def reset(self):
        self.update()

    def observe(self):
        return self.context, self.fc(self.context)

    def step(self, action):

        topk, _ = torch.topk(torch.tensor(self.reward), k=min(self.B, len(self.reward)))
        oracle = topk.sum().item()
        return self.reward[action], self.reward[action].sum(), oracle

    def update(self):

        n = poisson.rvs(mu=self.pmean)
        c2 = np.random.rand(n)
        index = np.random.randint(0, self.loc.shape[0], size=n)
        target_loc = np.random.rand(2).reshape((1,-1))
        c1 = np.linalg.norm(self.loc[index]-target_loc, axis=1)
        self.context = np.hstack((c1.reshape(-1,1), c2.reshape(-1,1)))
        self.reward = self.f(self.context) + self.sigma_f * np.random.randn(n)


