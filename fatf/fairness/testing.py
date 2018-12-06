# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 14:36:09 2018

@author: rp13102
"""
from __future__ import division
from sklearn.linear_model import LogisticRegression
import numpy as np

#from metrics import perform_checks_on_split, get_summary, counterfactual_fairness, individual_fairness, check_systemic_bias, check_sampling_bias, check_systematic_error
from metrics import FairnessChecks

def remove_field(dataset, field):
    field_names = list(dataset.dtype.names)
    if field in field_names:
        field_names.remove(field)
    return dataset[field_names]
    
testdata3 = np.array([
        ('Heidi Mitchell', 'uboyd@hotmail.com', 74, 52, 0, '0011', 1, '03/06/2018', 2),
       ('Tina Burns', 'stevenwheeler@williams.bi',  3, 86, 1, '0011', 0, '26/09/2017', 1),
       ('Justin Brown', 'velasquezjake@gmail.com', 3, 86, 0, '0011', 2, '31/12/2015', 0),
       ('Brent Parker', 'kennethsingh@strong-foley', 70, 57, 0, '0011', 1, '02/10/2011', 0),
       ('Bryan Norton', 'erica36@hotmail.com', 48, 57, 0, '1100', 0, '09/09/2012', 2),
       ('Ms. Erin Craig', 'ritterluke@gmail.com', 30, 98, 0, '1100', 2, '04/11/2006', 1),
       ('Gerald Park', 'larrylee@hayes-brown.net', 41, 73, 1, '1100', 0, '15/12/2015', 0),
       ],
      dtype=[('name', '<U16'), ('email', '<U25'), ('age', '<i4'), ('weight', '<i4'), ('gender', '<i4'), ('zipcode', '<U6'), ('target', '<i4'), ('dob', '<U10'), ('prediction', '<i4')])

def create_dataset():
    list_of_dictionaries = get_data()

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
    return dataset, treatments, distance_funcs

def get_dictionary(field_name, field_data, field_treatment, field_distance_func):
    dictionary =  {
                'name': field_name,
                'data': field_data,
                'treatment': field_treatment,
                'distance_func': field_distance_func
                }  
    
    return dictionary

def zipcode_dist(x, y):
    n = len(x)
    t = sum([item[0] == item[1] for item in zip(x, y)])
    return 1 - t/n
    
def get_data():
    age_dict = get_dictionary(field_name = 'Age', 
                              field_data = testdata3['age'], 
                              field_treatment = 'Feature', 
                              field_distance_func = lambda x, y: abs(x - y)
                              )
    
    weight_dict = get_dictionary(field_name = 'Weight', 
                              field_data = testdata3['weight'], 
                              field_treatment = 'Feature', 
                              field_distance_func = lambda x, y: abs(x - y)
                              )
    
    disease_dict = get_dictionary(field_name = 'Target', 
                              field_data = testdata3['target'],
                              field_treatment = 'Target', 
                              field_distance_func = lambda x, y: x == y
                              )


   
    zipcode_dict = get_dictionary(field_name = 'Zipcode', 
                              field_data = testdata3['zipcode'], 
                              field_treatment = 'Feature', 
                              field_distance_func = zipcode_dist
                              )
    
    gender_dict = get_dictionary(field_name = 'Gender', 
                              field_data = testdata3['gender'], 
                              field_treatment = 'Protected', 
                              field_distance_func = lambda x, y: x == y
                              )

    prediction_dict = get_dictionary(field_name = 'Prediction', 
                              field_data = testdata3['prediction'], 
                              field_treatment = 'ToIgnore', 
                              field_distance_func = None
                              )
    
    return [age_dict, weight_dict, disease_dict, zipcode_dict, gender_dict, prediction_dict]

def get_boundaries(field, increments=5):
    max_val = np.max(field)
    min_val = np.min(field)
    return np.linspace(min_val, max_val+1, increments)

dataset, treatments, distance_funcs = create_dataset()
targets = dataset['Target']
predictions = dataset['Prediction']
dataset = remove_field(dataset, 'Target')
dataset = remove_field(dataset, 'Prediction')


features_to_check = ['Gender']

mdl = FairnessChecks(dataset, 
                     targets,
                     distance_funcs,
                     protected = treatments['Protected'][0],
                     toignore = treatments['ToIgnore']
                     )

c=mdl.check_systemic_bias()


d=mdl.check_sampling_bias(features_to_check=features_to_check)


features_to_check = ['Age', 'Gender']
boundaries = {'Age': []}
for key, value in boundaries.items():
    if len(value) == 0:
        boundaries[key] = get_boundaries(dataset[key])
        
e=mdl.check_sampling_bias(features_to_check=features_to_check, 
                          return_weights = True, 
                          boundaries_for_numerical = boundaries)

f=mdl.check_systematic_error(predictions = predictions,
                             features_to_check=['Gender'],
                             requested_checks='all',
                             boundaries_for_numerical=boundaries)

aggregated_checks = mdl.perform_checks_on_split(
                                                get_summary = True,
                                                requested_checks=['accuracy'],
                                                conditioned_field='Zipcode',
                                                condition='1100')

aggregated_checks2 = mdl.perform_checks_on_split(
                                                get_summary = True,
                                                requested_checks=['accuracy'])

X = remove_field(dataset, 'Zipcode')

model = LogisticRegression()

newx = np.array(X.tolist())
model.fit(newx, targets)


cm = mdl.counterfactual_fairness(model, 'Gender', X, [0, 1, 2])


g = mdl.individual_fairness(model, newx)

