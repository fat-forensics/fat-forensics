# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 10:19:48 2018

@author: Rafael Poyiadzi
"""
import math
import datetime
import numpy as np

#from fatf.accountability.data.supp import testdata3
import numpy as np
import itertools

attributes = ['name',
              'email',
              'age',
              'weight',
              'gender',
              'zipcode',
              'diagnosis',
              'diagnosis_2']

attributes2 = ['name',
              'email',
              'age',
              'weight',
              'gender',
              'zipcode',
              'diagnosis',
              'diagnosis_2']

sensitive_attributes_dicts = {
        'diagnosis': ['cancer', 'heart', 'lung', 'hip']
        }

sensitive_attributes2 = {
        'diagnosis': ['cancer', 'heart', 'lung', 'hip'],
        'diagnosis_2': ['A', 'B', 'C', 'D']
        }

quasi_identifiers_dicts = {
        'age' : 0,
        'weight': 0,
        'gender': ['male', 'female'],
        'zipcode': ['100', '101', '110', '111']}

maxima = {
        'age': 100,
        'weight': 100,
        'gender': 1,
        'zipcode': 6}

testdata3 = np.array([('Heidi Mitchell', 'uboyd@hotmail.com', 74, 52, 'female', '1121', 'cancer', '03/06/2018'),
       ('Tina Burns', 'stevenwheeler@williams.bi',  3, 86, 'female', '0323', 'hip', '26/09/2017'),
       ('Justin Brown', 'velasquezjake@gmail.com', 26, 56, 'female', '0100', 'heart', '26/12/2015'),
       ('Brent Parker', 'kennethsingh@strong-foley', 70, 57, 'male', '3131', 'heart', '02/10/2011'),
       ('Bryan Norton', 'erica36@hotmail.com', 48, 57, 'male', '0301', 'hip', '09/09/2012'),
       ('Ms. Erin Craig', 'ritterluke@gmail.com', 30, 98, 'male', '2223', 'cancer', '04/11/2006'),
       ('Gerald Park', 'larrylee@hayes-brown.net', 41, 73, 'female', '0101', 'heart', '15/12/2015'),
       ('Monica Fry', 'morenocraig@howard.com', 24,  1, 'male', '1212', 'hip', '21/12/2005'),
       ('Michael Smith', 'edward72@dunlap-jackson.c', 44, 66, 'male', '0111', 'hip', '07/11/2012'),
       ('Dean Campbell', 'michele18@hotmail.com', 62, 96, 'female', '2320', 'lung', '22/01/2009'),
       ('Kimberly Kent', 'wilsoncarla@mitchell-gree', 63, 51, 'male', '2003', 'cancer', '16/06/2017'),
       ('Michael Burnett', 'collin04@scott.org', 26, 88, 'male', '0301', 'heart', '07/03/2009'),
       ('Patricia Richard', 'deniserodriguez@hotmail.c', 94, 64, 'female', '3310', 'heart', '20/08/2006'),
       ('Joshua Ramos', 'michaelolson@yahoo.com', 59, 19, 'female', '3013', 'cancer', '22/07/2005'),
       ('Samuel Fletcher', 'jessicagarcia@hotmail.com', 14, 88, 'female', '1211', 'lung', '29/07/2004'),
       ('Donald Hess', 'rking@gray-mueller.com', 16, 15, 'male', '0102', 'hip', '16/09/2010'),
       ('Rebecca Thomas', 'alex57@gmail.com', 94, 48, 'female', '0223', 'cancer', '05/02/2000'),
       ('Hannah Osborne', 'ericsullivan@austin.com', 41, 25, 'female', '0212', 'heart', '11/06/2012'),
       ('Sarah Nelson', 'davidcruz@hood-mathews.co', 36, 57, 'female', '0130', 'cancer', '13/01/2003'),
       ('Angela Kelly', 'pwilson@howell-bryant.com', 37, 52, 'female', '1023', 'heart', '28/03/2009'),
       ('Susan Williams', 'smithjoshua@allen.com', 21, 42, 'male', '0203', 'lung', '15/11/2005')],
      dtype=[('name', '<U16'), ('email', '<U25'), ('age', '<i4'), ('weight', '<i4'), ('gender', '<U10'), ('zipcode', '<U6'), ('diagnosis', '<U6'), ('dob', '<U16')])

testdata = np.array([('Brandon Liu', 'adamsdiane@gmail.com', 91, 16, 'male', '0123', 'hip', '21/01/2009'),
       ('Lori Erickson', 'millerclarence@gmail.com', 27, 62, 'male', '2312', 'heart', '10/01/2000'),
       ('Carolyn Adams', 'sethellis@smith.com', 78, 53, 'male', '3033', 'cancer', '18/12/2006'),
       ('Donna Orr', 'jgibson@hunter.com', 50, 81, 'female', '1013', 'cancer', '14/01/2002'),
       ('Jennifer Cook', 'ksingleton@brown.com', 64, 10, 'male', '0000', 'cancer', '12/10/2004'),
       ('Kara Cunningham ', 'strongbrittany@gmail.com', 88,  6, 'female', '1013', 'lung', '28/03/2015'),
       ('Phillip Saunders', 'kruiz@gmail.com', 20, 19, 'female', '2133', 'lung', '29/10/2019'),
       ('Carrie Park', 'wendy11@daniel.org', 51, 77, 'female', '3313', 'hip', '30/07/2002'),
       ('Mr. John Thompso', 'wglenn@smith.com', 55, 86, 'female', '1300', 'cancer', '20/05/2007'),
       ('Shane Hall', 'munozmatthew@smith.org', 41, 51, 'female', '2212', 'heart', '18/08/2005'),
       ('Willie Miller', 'villarrealbenjamin@wilson', 54, 27, 'female', '0212', 'heart', '15/10/2015')],
      dtype=[('name', '<U16'), ('email', '<U25'), ('age', '<i4'), ('weight', '<i4'), ('gender', '<U6'), ('zipcode', '<U6'), ('diagnosis', '<U6'), ('dob', '<U10')])

testdata4 = np.array([('Brandon Liu', 'adamsdiane@gmail.com', 91, 16, 'male', '0123', 'hip', '21/01/2009'),
       ('Lori Erickson', 'millerclarence@gmail.com', 27, 62, 'male', '2312', 'heart', '10/01/2000'),
       ('Carolyn Adams', 'sethellis@smith.com', 78, 53, 'male', '3033', 'cancer', '18/12/2006'),
       ('Donna Orr', 'jgibson@hunter.com', 50, 81, 'female', '1013', 'cancer', '14/01/2002'),
       ('Jennifer Cook', 'ksingleton@brown.com', 64, 10, 'male', '0000', 'cancer', '12/10/2004'),
       ('Kara Cunningham ', 'strongbrittany@gmail.com', 88,  6, 'female', '1013', 'lung', '28/03/2005'),],
      dtype=[('name', '<U16'), ('email', '<U25'), ('age', '<i4'), ('weight', '<i4'), ('gender', '<U6'), ('zipcode', '<U6'), ('diagnosis', '<U6'), ('dob', '<U10')])

input_dict = {'data': testdata,
              #'attributes': testdata.dtype.names,
              'sensitive_attributes': sensitive_attributes_dicts.keys(),
              'quasi_identifiers': quasi_identifiers_dicts.keys()}

testdata2 = np.array([('Megan King DDS', 'jeromelane@smith.com', 39, 16, 'female', '000', 'lung', 'D'),
       ('Elizabeth Chang', 'jnichols@hunt.com', 49, 99, 'female', '000', 'cancer', 'B'),
       ('Marcus Rios', 'hholmes@yahoo.com', 50, 63, 'male', '001', 'heart', 'D'),
       ('Jonathan Davis', 'james53@hotmail.com', 56, 46, 'male', '000', 'heart', 'A'),
       ('Anthony Miles', 'renee40@hotmail.com', 81, 80, 'female', '001', 'hip', 'A'),
       ('Steven Medina', 'nathanieldunn@parker.com', 29, 98, 'male', '000', 'hip', 'C'),
       ('Paula Gallegos', 'cheryl30@yahoo.com', 58, 92, 'female', '001', 'cancer', 'B'),
       ('Alex Torres', 'jamesdouglas@marshall.inf', 39, 84, 'male', '000', 'heart', 'C'),
       ('Stacie Davis DDS', 'melissa05@frazier-hernand', 89, 52, 'male', '000', 'cancer', 'C'),
       ('Mr. Michael Step', 'brad62@estes-hernandez.co', 59, 44, 'male', '000', 'heart', 'A'),
       ('Andrew Wright', 'stephaniediaz@gmail.com', 63, 31, 'male', '000', 'cancer', 'C')],
      dtype=[('name', '<U16'), ('email', '<U25'), ('age', '<i4'), ('weight', '<i4'), ('gender', '<U6'), ('zipcode', '<U3'), ('diagnosis', '<U6'), ('diagnosis_2', '<U6')])

# =============================================================================
# input_dict2 = {'data': testdata2,
#               'attributes': attributes2,
#               'sensitive_attributes': sensitive_attributes2,
#               'quasi_identifiers': quasi_identifiers}
# =============================================================================

expected_testdata2 = np.array([
       ('Megan King DDS', 'jeromelane@smith.com', 39, 16, 'female', '000', 'lung', 'D', 'lungD'),
       ('Elizabeth Chang', 'jnichols@hunt.com', 49, 99, 'female', '000', 'cancer', 'B', 'cancerB'),
       ('Marcus Rios', 'hholmes@yahoo.com', 50, 63, 'male', '001', 'heart', 'D', 'heartD'),
       ('Jonathan Davis', 'james53@hotmail.com', 56, 46, 'male', '000', 'heart', 'A', 'heartA'),
       ('Anthony Miles', 'renee40@hotmail.com', 81, 80, 'female', '001', 'hip', 'A', 'hipA'),
       ('Steven Medina', 'nathanieldunn@parker.com', 29, 98, 'male', '000', 'hip', 'C', 'hipC'),
       ('Paula Gallegos', 'cheryl30@yahoo.com', 58, 92, 'female', '001', 'cancer', 'B', 'cancerB'),
       ('Alex Torres', 'jamesdouglas@marshall.inf', 39, 84, 'male', '000', 'heart', 'C', 'heartC'),
       ('Stacie Davis DDS', 'melissa05@frazier-hernand', 89, 52, 'male', '000', 'cancer', 'C', 'cancerC'),
       ('Mr. Michael Step', 'brad62@estes-hernandez.co', 59, 44, 'male', '000', 'heart', 'A', 'heartA'),
       ('Andrew Wright', 'stephaniediaz@gmail.com', 63, 31, 'male', '000', 'cancer', 'C', 'cancerC')],
      dtype=[('name', '<U16'), ('email', '<U25'), ('age', '<i4'), ('weight', '<i4'), ('gender', '<U6'), ('zipcode', '<U3'), ('diagnosis', '<U6'), ('diagnosis_2', '<U6'), ('new_sa', '<U7')])

expected_sensitive_attributes2 = {
        'diagnosis': ['cancer', 'heart', 'lung', 'hip'],
        'diagnosis_2': ['A', 'B', 'C', 'D'],
        'new_sa': np.array(sorted(list(map(lambda x: x[0] + x[1], 
                           list(itertools.product(['cancer', 'heart', 'lung', 'hip'], 
                                                  ['A', 'B', 'C', 'D']))))))
        }

expected_attributes2 = ['name',
              'email',
              'age',
              'weight',
              'gender',
              'zipcode',
              'diagnosis',
              'diagnosis_2',
              'new_sa']

expected_dict2 = {'data': expected_testdata2,
                  'attributes': expected_attributes2,
                  'sensitive_attributes': expected_sensitive_attributes2,
                  'quasi_identifiers': quasi_identifiers_dicts.keys()}

import datetime
dates = [datetime.date(2002, 11, 17),
 datetime.date(1972, 9, 26),
 datetime.date(2007, 8, 10),
 datetime.date(2005, 3, 15),
 datetime.date(1972, 12, 21),
 datetime.date(2014, 9, 25),
 datetime.date(2015, 6, 5),
 datetime.date(2005, 3, 8),
 datetime.date(1972, 10, 23),
 datetime.date(1999, 7, 20)]


def remove_field(dataset, field):
    field_names = list(dataset.dtype.names)
    if field in field_names:
        field_names.remove(field)
    return dataset[field_names]

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
        if (item < bounds[0] or item > bounds[1] + 1):
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
                   (21, 32)]
    
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
                   (21, 32)]
    
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
                    return generate_date(years[0], months[0])
            else:
                return generate_date(years[0], months[0])

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

from new import TCloseness, LDiversity, KAnonymity  
dataset, treatments, lca_funcs, distance_funcs, range_funcs = create_dataset()  
#treatments['QI'].pop()
#treatments['SA'].append('Gender')  
del lca_funcs['Age']
mdl0 = TCloseness(dataset, treatments['I'], treatments['QI'], treatments['SA'], 
                 lca_funcs=lca_funcs, range_funcs=range_funcs, t=0.30) 
d=mdl0.apply_tcloseness_2(suppress=True)
print(mdl0.emds)
mdl1 = TCloseness(dataset, treatments['I'], treatments['QI'], treatments['SA'], 
                 lca_funcs=lca_funcs, range_funcs=range_funcs, t=0.30) 
e=mdl1.apply_tcloseness(suppress=True)
print(mdl1.emds)

# =============================================================================
# mdl = LDiversity(dataset, treatments['I'], treatments['QI'], treatments['SA'], 
#                  lca_funcs=lca_funcs, range_funcs=range_funcs, l=2)
# a=mdl.apply_ldiversity(suppress=True)
# 
# mdl = KAnonymity(dataset, treatments['I'], treatments['QI'], treatments['SA'], 
#                  lca_funcs=lca_funcs, range_funcs=range_funcs, k=2)
# c=mdl.apply_kanonymity(suppress=True)
# 
# =============================================================================
