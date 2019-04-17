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


    dataset, targets, treatments, distance_funcs = create_dataset()

    model = LogisticRegression(solver = 'lbfgs')

    newx = dataset.copy(order='K')
    newx = np.array(newx.tolist())
    model.fit(newx, targets)

    categorical_features = ['job_type']

    del distance_funcs['Target']


    dataset.astype(dtype=[('Age', '<f4'), ('Weight', '<f4'), ('job_type', '<f4')])
    x0 = dataset[0]

    pred = model.predict(np.array(x0.tolist(), dtype=float).reshape(1, -1))
    feature_ranges = {
                        'Age': {'max': x0['Age'] + 10,
                                'min': x0['Age'] - 10},
                      'job_type': np.array([x0['job_type']])
                      }

    stepsizes = {
                'Age': 0.50,
                'Weight': 0.50
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
    c=mdl.generate_counterfactual(x0, target_class)


def create_dataset():
    import numpy as np
    testdata3 = np.array([
            ('Heidi Mitchell', 'uboyd@hotmail.com', 35, 52, 2, '0011', 1, '03/06/2018', 1),
           ('Tina Burns', 'stevenwheeler@williams.bi',  3, 86, 1, '0011', 0, '26/09/2017', 1),
           ('Justin Brown', 'velasquezjake@gmail.com', 3, 86, 2, '0011', 1, '31/12/2015', 0),
           ('Brent Parker', 'kennethsingh@strong-foley', 70, 57, 0, '0011', 1, '02/10/2011', 0),
           ('Bryan Norton', 'erica36@hotmail.com', 48, 57, 3, '1100', 0, '09/09/2012', 1),
           ('Ms. Erin Craig', 'ritterluke@gmail.com', 30, 98, 0, '1100', 1, '04/11/2006', 1),
           ('Gerald Park', 'larrylee@hayes-brown.net', 41, 73, 1, '1100', 0, '15/12/2015', 0),
           ],
          dtype=[('name', '<U16'), ('email', '<U25'), ('age', '<f4'),
                 ('weight', '<f4'), ('job_type', '<i4'), ('zipcode', '<U6'),
                 ('target', '<i4'), ('dob', '<U10'), ('prediction', '<i4')])
    age_dict = {'name' : 'Age',
                'data' : testdata3['age'],
                'treatment' : 'Feature',
                'distance_func' : lambda x, y: abs(x - y)}
    weight_dict = {'name' : 'Weight',
                   'data' : testdata3['weight'],
                   'treatment' : 'Feature',
                   'distance_func' : lambda x, y: abs(x - y)}
    disease_dict = {'name' : 'Target',
                    'data' : testdata3['target'],
                    'treatment' : 'Target',
                    'distance_func' : lambda x, y: x == y}
    zipcode_dict = {'name' : 'Zipcode',
                    'data' : testdata3['zipcode'],
                    'treatment' : 'Feature',
                    'distance_func' : lambda x, y: 1 - sum([item[0] == item[1] for item in zip(x, y)])/len(x)}
    job_type_dict = {'name' : 'job_type',
                     'data' : testdata3['job_type'],
                     'treatment' : 'Protected',
                     'distance_func' : lambda x, y: int(x != y)}
    prediction_dict = {'name' : 'Prediction',
                       'data' : testdata3['prediction'],
                       'treatment' : 'ToIgnore',
                       'distance_func' : None}
    list_of_dictionaries = [age_dict, weight_dict, disease_dict, job_type_dict]




    desired_keys = ['name',
                    'data',
                    'treatment',
                    'distance_func'
                    ]
    dts = []
    treatments = {
                'Protected': [],
                'Feature': [],
                'ToIgnore': [],
                'Target': []
                }
    distance_funcs = {}
    data = []


    for dictionary in list_of_dictionaries:
        current_dictionary_keys = dictionary.keys()
        for key in desired_keys:
            if key not in current_dictionary_keys:
                raise ValueError('One of the provided dictionaries does not have the key: ' + str(key))

        field_name = dictionary['name']
        field_col = dictionary['data']
        if type(field_col) != np.ndarray:
            raise TypeError(str(field_name) + ' data should be of type numpy.ndarray.')

        data.append(field_col)

        dts.append((field_name, field_col.dtype))
        distance_funcs[field_name] = dictionary['distance_func']

        field_treatment = dictionary['treatment']

        if field_treatment == 'Protected':
            treatments['Protected'].append(field_name)
        elif field_treatment == 'Feature':
            treatments['Feature'].append(field_name)
        elif field_treatment == 'Target':
            treatments['Target'].append(field_name)
        elif field_treatment == 'ToIgnore':
            treatments['ToIgnore'].append(field_name)
        else:
            raise ValueError('Unknown treatment')

    N = data[0].shape[0]
    if not np.all(column.shape[0] == N for column in data):
        raise ValueError('Data provided is of different length.')

    dataset = np.array([item for item in zip(*data)], dtype=dts)

    targets = dataset['Target']
    # remove targets form the dataset
    field_names = list(dataset.dtype.names)
    field_names.remove('Target')
    dataset = dataset[field_names]

    return dataset, targets, treatments, distance_funcs
