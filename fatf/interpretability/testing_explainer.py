# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 11:52:13 2018

@author: rp13102
"""
import numpy as np
from cf_explainer import Explainer
from sklearn.linear_model import LogisticRegression
from supp_explainer import create_dataset

class CostObject(object):
    def __init__(self, distance_funcs):
        self.distance_funcs = distance_funcs
        
    """
    Computes the distance between two instances, based on the distance functions
    provided by the user.
    """
    def dist(self, v0, v1):
        dist = 0
        features = v0.dtype.names
        for feature in features:
            dist += distance_funcs[feature](v0[feature], v1[feature])
        return dist

def remove_field(dataset, field):
    field_names = list(dataset.dtype.names)
    if field in field_names:
        field_names.remove(field)
    return dataset[field_names]

dataset, treatments, distance_funcs = create_dataset()
targets = dataset['Target']
dataset = remove_field(dataset, 'Target')

model = LogisticRegression(solver = 'lbfgs')

newx = dataset.copy(order='K')
newx = np.array(newx.tolist())
model.fit(newx, targets)

categorical_features = ['Gender']

del distance_funcs['Target']



#cost_calc = CostObject(distance_funcs)
dataset.astype(dtype=[('Age', '<f4'), ('Weight', '<f4'), ('Gender', '<f4')])
x0 = dataset[0]
print(x0)
pred = model.predict(np.array(x0.tolist(), dtype=float).reshape(1, -1))
feature_ranges = {
                    'Age': {'max': x0['Age']
                            },
                  'Gender': np.array([x0['Gender']])
                  }

stepsizes = {
            'Age': 0.10,
            'Weight': 0.10
            }


mdl = Explainer(model, 
                categorical_features, 
                dataset,
                monotone = False,
                stepsizes = stepsizes,
                feature_ranges=feature_ranges,
                max_comb = 2,
                dist_funcs = distance_funcs
                )

target_class = 0
import time
start = time.time()
c=mdl.generate_counterfactual(x0,
                            target_class
                            )
end = time.time()
print(end - start)

