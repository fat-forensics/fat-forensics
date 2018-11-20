# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 14:36:09 2018

@author: rp13102
"""
from sklearn.linear_model import LogisticRegression

from supp_data import generatetable
from metrics import perform_checks_on_split, get_summary, counterfactual_fairness, individual_fairness

def euc_dist(v0, v1):
    return np.linalg.norm(v0 - v1)**2

def remove_field(dataset, field):
    field_names = list(dataset.dtype.names)
    if field in field_names:
        field_names.remove(field)
    return dataset[field_names]

checks = {'accuracy': lambda x: sum(np.diag(x)) / np.sum(x),
          'true_positives': lambda x: x[0, 0],
          'true_negatives': lambda x: x[1, 1],
          'false_positives': lambda x: x[0, 1],
          'false_negatives': lambda x: x[1, 0],
          'true_positive_rate': lambda x: x[0, 0] / (x[0, 0] + x[0, 1]),
          'true_negative_rate': lambda x: x[1, 1] / (x[1, 1] + x[1, 0]),
          'false_positive_rate': lambda x: x[0, 1] / (x[0, 0] + x[0, 1]),
          'false_negative_rate': lambda x: x[1, 1] / (x[1, 1] + x[1, 0]),
          'treatment': lambda x: x[0, 1] / x[1, 0]
          }


testdata = generatetable()

targets = testdata['target']
predictions = testdata['prediction']
X = remove_field(testdata, 'target')
X = remove_field(testdata, 'predictions')


aggregated_checks = perform_checks_on_split(X, targets, predictions, 'gender', checks, 'feature1', 0)
summary = get_summary(aggregated_checks)


model = LogisticRegression()
cm = counterfactual_fairness(model, X, targets, 'gender')

newx = np.array(X.tolist())
model.fit(newx, y)
predictions_proba = model.predict_proba(newx)
fair_bool = individual_fairness(newx, predictions_proba, euc_dist, euc_dist)
