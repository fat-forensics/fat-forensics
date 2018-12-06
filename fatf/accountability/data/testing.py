# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 10:19:48 2018

@author: Rafael Poyiadzi
"""
import math
import datetime
import numpy as np

from supp import testdata3


def create_dataset():
    list_of_dictionaries = get_data()

    desired_keys = ['name',
                    'data',
                    'treatment',
                    'lca_func',
                    'range_func',
                    'distance_func']
    
    dts = []
    treatments = {
                'I': [],
                'SA': [],
                'QI': []
                }
    distance_funcs = {}
    lca_funcs = {}
    data = []
    
    range_funcs = {}
    
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
        
        current_field_lca_func = dictionary['lca_func']
        lca_funcs[field_name] = current_field_lca_func
        
        current_field_range_func = dictionary['range_func']
        range_funcs[field_name] = current_field_range_func

        field_treatment = dictionary['treatment']
        
        if field_treatment == 'I':
            treatments['I'].append(field_name)
        elif field_treatment == 'SA':
            treatments['SA'].append(field_name)
        elif field_treatment == 'QI':
            treatments['QI'].append(field_name)
            if not current_field_lca_func:
                raise ValueError(str(field_name) + ' field requires an LCA function.')
            if not current_field_range_func:
                raise ValueError(str(field_name) + ' field requires a range function.')
        else:
            raise ValueError('Unknown treatment')
            
    N = data[0].shape[0]
    if not np.all(column.shape[0] == N for column in data):
        raise ValueError('Data provided is of different length.')
        
    dataset = np.array([item for item in zip(*data)], dtype=dts)
    return dataset, treatments, lca_funcs, distance_funcs, range_funcs


def check_interval(input_list, bounds):
    for item in input_list:
        if (item < bounds[0] or item > bounds[1]):
            return False
    return True

def lca_realline(X, rounding=5):
    X = list(map(int, X))
    max_val = max(X)
    min_val = min(X)
    lb = math.floor(int(min_val/rounding)*rounding)
    ub = math.ceil(int(max_val/rounding)*rounding)
    lca = str(lb) + '-' + str(ub)
    
    return lca 

def get_lca_zipcode(X):
    while len(set(X)) != 1:
        X = list(map(generalize_string, X))
    return X[0]

def generalize_string(x):
    """ 
        Generalizes the last element of a string to '*'.
    """
    x = list(str(x))
    for idx, item in enumerate(x[::-1]):
        if item == '*':
            continue
        else:
            x[-(idx+1)] = '*'
            break
    return "".join(x)  

def range_func_realline(X):
    X = list(map(int, X))
    max_val = max(X)
    min_val = min(X)
    return max_val - min_val

def get_depthoftree_dob(attribute):
    if len(set(attribute)) == 1:
        return 0
    
    days_bounds = [(1, 10),
                   (11, 20),
                   (21, 31)]
    
    months_bounds = [(1, 3),
                    (4, 6),
                    (7, 9),
                    (10, 12)]
    
    dates_list = list(map(lambda x: datetime.datetime.strptime(x, '%d/%m/%Y'), attribute))
    extracted_dates = [(item.day, item.month, item.year) for item in dates_list]
    years = [str(item[2]) for item in extracted_dates]
    for idx in range(4):
        l = [item[idx] for item in years]
        if len(set(l)) != 1:
            break

    months = [item[1] for item in extracted_dates]
    for m_bounds in months_bounds:
        if check_interval(months, m_bounds):
            idx += 1
            if len(set(months)) == 1:
                idx += 1
            
            days = [item[0] for item in extracted_dates]
            for d_bounds in days_bounds:
                if check_interval(days, d_bounds):
                    idx += 1
                    if len(set(days)) == 1:
                        idx += 1
                    break
    return 8 - idx

def generate_date(year='****', month='**', day='**'):
    return str(day) + '/' + str(month) + '/' + str(year)
              
def get_lca_dates(dates_list):
    dates_list = list(map(lambda x: datetime.datetime.strptime(x, '%d/%m/%Y'), dates_list))
    extracted_dates = [(item.day, item.month, item.year) for item in dates_list]
    years = [str(item[2]) for item in extracted_dates]
    
    days_bounds = [(1, 10),
                   (11, 20),
                   (21, 31)]
    
    months_bounds = [(1, 3),
                    (4, 6),
                    (7, 9),
                    (10, 12)]
    
    if len(set(years)) > 1:
        while len(set(years)) > 1:
            years = list(map(generalize_string, years))
        return generate_date(years[0])
    else:
        months = [item[1] for item in extracted_dates] 
        if len(set(months)) > 1:
            for bounds in months_bounds:
                if check_interval(months, bounds):
                    month = str(bounds[0]) + '-' + str(bounds[1])
                    return generate_date(years[0], month)
            return generate_date(years[0])
        else:
            days = [item[0] for item in extracted_dates]
            if len(set(days)) > 1:
                for bounds in days_bounds:
                    if check_interval(days, bounds):
                        day = str(bounds[0]) + '-' + str(bounds[1])
                        return generate_date(years[0], months[0], day)
            else:
                return generate_date(years[0], months[0], days[0])

def get_depthoftree_zipcode(attribute):
    p = len(attribute[0])
    if len(set(attribute)) == 1:
        return 0
    
    for idx in range(p):
        l = [item[idx] for item in attribute]
        if len(set(l)) != 1:
            break
    return p - idx

def get_dictionary(field_name, field_data, field_treatment, field_distance_func, field_lca_func, field_range_func):
    dictionary =  {
                'name': field_name,
                'data': field_data,
                'treatment': field_treatment,
                'distance_func': field_distance_func,
                'lca_func': field_lca_func,
                'range_func': field_range_func
                }  
    
    return dictionary

def get_lca_gender(X):
    if len(set(X)) == 1:
        return X[0]
    else:
        return 'Gender'
    
def get_depthoftree_gender(X):
    if len(set(X)) == 1:
        return 0
    else:
        return 1
    
def get_data():
    age_dict = get_dictionary(field_name = 'Age', 
                              field_data = testdata3['age'], 
                              field_treatment = 'QI', 
                              field_distance_func = lambda x, y: abs(x - y), 
                              field_lca_func = lca_realline, 
                              field_range_func = range_func_realline)

    weight_dict = get_dictionary(field_name = 'Weight', 
                              field_data = testdata3['weight'], 
                              field_treatment = 'QI', 
                              field_distance_func = lambda x, y: abs(x - y), 
                              field_lca_func = lca_realline, 
                              field_range_func = range_func_realline)

    name_dict = get_dictionary(field_name = 'Name', 
                              field_data = testdata3['name'], 
                              field_treatment = 'I', 
                              field_distance_func = None, 
                              field_lca_func = None, 
                              field_range_func = None)
    
    disease_dict = get_dictionary(field_name = 'Disease', 
                              field_data = testdata3['diagnosis'],
                              field_treatment = 'SA', 
                              field_distance_func = None, 
                              field_lca_func = None, 
                              field_range_func = None)

    dob_dict = get_dictionary(field_name = 'Dob', 
                              field_data = testdata3['dob'],
                              field_treatment = 'QI', 
                              field_distance_func = None, 
                              field_lca_func = get_lca_dates, 
                              field_range_func = get_depthoftree_dob)
    

    zipcode_dict = get_dictionary(field_name = 'Zipcode', 
                              field_data = testdata3['zipcode'], 
                              field_treatment = 'QI', 
                              field_distance_func = None, 
                              field_lca_func = get_lca_zipcode, 
                              field_range_func = get_depthoftree_zipcode)
    
    gender_dict = get_dictionary(field_name = 'Gender', 
                              field_data = testdata3['gender'], 
                              field_treatment = 'QI', 
                              field_distance_func = None, 
                              field_lca_func = get_lca_gender, 
                              field_range_func = get_depthoftree_gender)
    
    email_dict = get_dictionary(field_name = 'Email', 
                              field_data = testdata3['email'], 
                              field_treatment = 'I', 
                              field_distance_func = None, 
                              field_lca_func = None, 
                              field_range_func = None)
    
    return [name_dict, age_dict, weight_dict, disease_dict, dob_dict, zipcode_dict, gender_dict, email_dict]

