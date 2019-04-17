# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 17:47:33 2018

@author: rp13102
"""
import pytest
import numpy as np
from fatf.transparency.predictions.counterfactuals import Explainer, combine_arrays

input_array0_0 = np.array([1,2,3])
input_array1_0 = np.array([4,5,6,7,8])
expected_output_0 = [1,4,2,5,3,6,7,8]

@pytest.mark.parametrize("array0, array1, expected_output",
                         [(input_array0_0, input_array1_0, expected_output_0)])
def test_combine_arrays(array0, array1, expected_output):
    output_array = combine_arrays(array0, array1)
    assert np.all(output_array == expected_output)


def test_counterfactual():
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    data = np.array([
            ('Heidi Mitchell', 'uboyd@hotmail.com', 35, 52, 2, '0011', '03/06/2018', 1),
           ('Tina Burns', 'stevenwheeler@williams.bi',  3, 86, 1, '0011', '26/09/2017', 1),
           ('Justin Brown', 'velasquezjake@gmail.com', 3, 86, 2, '0011', '31/12/2015', 0),
           ('Brent Parker', 'kennethsingh@strong-foley', 70, 57, 0, '0011', '02/10/2011', 0),
           ('Bryan Norton', 'erica36@hotmail.com', 48, 57, 3, '1100', '09/09/2012', 1),
           ('Ms. Erin Craig', 'ritterluke@gmail.com', 30, 98, 0, '1100', '04/11/2006', 1),
           ('Gerald Park', 'larrylee@hayes-brown.net', 41, 73, 1, '1100', '15/12/2015', 0),
           ],
          dtype=[('name', '<U16'), ('email', '<U25'), ('age', '<f4'),
                 ('weight', '<f4'), ('job_type', '<i4'), ('zipcode', '<U6'),
                 ('dob', '<U10'), ('prediction', '<i4')])
    targets = np.array([1, 0, 1, 1, 0, 1, 0])
    targets_name = 'target'
    distance_functions = {
        'age': lambda x, y: abs(x - y),
        'weight': lambda x, y: abs(x - y),
        'target': lambda x, y: x == y,
        'zipcode': lambda x, y: 1 - sum([item[0] == item[1] for item in zip(x, y)])/len(x),
        'job_type': lambda x, y: int(x != y),
        'prediction': None
    }

    selected_columns = ['age', 'weight', 'job_type']

    dataset = data[selected_columns]
    targets = targets
    distance_funcs = {key: distance_functions[key] for key in selected_columns}




    dataset_unstructured = fuar.as_unstructured(dataset)
    clf = LogisticRegression(solver='lbfgs')
    clf.fit(dataset_unstructured, targets)

    categorical_features = ['job_type']

    stepsizes = {'age': 0.50,
                 'weight': 0.50}


    datapoint = dataset[0]
    datapoint_class = 0

    feature_ranges = {'age': {'max': datapoint['age'] + 10,
                              'min': datapoint['age'] - 10},
                      'job_type': np.array([datapoint['job_type']])
                      }


    mdl = Explainer(clf,
                    categorical_features,
                    dataset,
                    monotone = False,
                    stepsizes = stepsizes,
                    feature_ranges=feature_ranges,
                    max_comb = 2,
                    dist_funcs = distance_funcs
                    )
    c=mdl.generate_counterfactual(datapoint, datapoint_class)
