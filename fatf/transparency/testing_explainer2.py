# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 11:52:13 2018

@author: rp13102
"""
import numpy as np
from gradient_explainer import Explainer2
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

n_ftrs = 2
X, y = make_classification(n_features=n_ftrs, n_redundant=0, n_informative=1,
                             n_clusters_per_class=1)
X = X - np.mean(X, axis = 0)
X = X / np.std(X, axis = 0)
mdl = LogisticRegression(solver = 'lbfgs')
mdl.fit(X, y)
pr_proba = mdl.predict_proba(X)
pr = mdl.predict(X)
np.random.seed(42)
min_values = np.min(X, axis=0)
max_values = np.max(X, axis=0)
boundaries = {}
for i in range(n_ftrs):
    boundaries[i] = {'min': min_values[i],
                      'max': max_values[i]}

if pr[0] == 0:
    target = np.array([0, 1])
else:
    target = np.array([1, 0])

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=pr_proba[:, 1])
plt.scatter(X[0, 0], X[0, 1], marker='x')
#plt.colorbar()
xpl = Explainer2(mdl,
                 reg = 5,
                 )
newx = xpl.explain(X[0, :],
                   target,
                   nottochange=[1])

if newx is not 0:
    print(mdl.predict_proba(newx.reshape(1, -1)), target)
    print(newx - X[0, :])
