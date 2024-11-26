# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 10:06:23 2018

@author: rp13102
"""
from fatf.accountability.data.new import BaseAnonymiser

import pytest
import numpy as np
from fatf.accountability.data.supp_testing import create_dataset

dataset, treatments, lca_funcs, distance_funcs, range_funcs = create_dataset()

input0 = np.array([('Brandon Liu', 'adamsdiane@gmail.com', 91, 16, 'male', '0123', 'hip', '21/01/2009'),
       ('Lori Erickson', 'millerclarence@gmail.com', 27, 62, 'male', '2312', 'heart', '10/01/2000'),
       ('Carolyn Adams', 'sethellis@smith.com', 78, 53, 'male', '3033', 'cancer', '18/12/2006'),
       ('Donna Orr', 'jgibson@hunter.com', 50, 81, 'female', '1013', 'cancer', '14/01/2002'),
       ('Jennifer Cook', 'ksingleton@brown.com', 64, 10, 'male', '0000', 'cancer', '12/10/2004'),
       ('Kara Cunningham ', 'strongbrittany@gmail.com', 88,  6, 'female', '1013', 'lung', '28/03/2005'),],
      dtype=[('Name', '<U16'), ('Email', '<U25'), ('Age', '<i4'), 
             ('Weight', '<i4'), ('Gender', '<U6'), ('Zipcode', '<U6'), 
             ('Diagnosis', '<U6'), ('Dob', '<U10')])

output0 = np.array([('*', '*', 91, 16, 'male', '0123', 'hip', '21/01/2009'),
       ('*', '*', 27, 62, 'male', '2312', 'heart', '10/01/2000'),
       ('*', '*', 78, 53, 'male', '3033', 'cancer', '18/12/2006'),
       ('*', '*', 50, 81, 'female', '1013', 'cancer', '14/01/2002'),
       ('*', '*', 64, 10, 'male', '0000', 'cancer', '12/10/2004'),
       ('* ', '*', 88,  6, 'female', '1013', 'lung', '28/03/2005'),],
        dtype=[('Name', '<U16'), ('Email', '<U25'), ('Age', '<i4'), 
             ('Weight', '<i4'), ('Gender', '<U6'), ('Zipcode', '<U6'), 
             ('Diagnosis', '<U6'), ('Dob', '<U10')])

identifiers = ['Name', 'Email']
quasi_identifiers = ['Age', 'Weight', 'Gender', 'Zipcode', 'Dob']
sensitive = ['Diagnosis']

@pytest.mark.parametrize("input_dataset, identifiers, quasi_identifiers, sensitive, expected_output_dataset", 
                         [(input0, identifiers, quasi_identifiers, sensitive, output0)])
def test_suppress(input_dataset, identifiers, quasi_identifiers, sensitive, 
                  expected_output_dataset):
    
    mdl = BaseAnonymiser(input_dataset, 
                     identifiers,
                     quasi_identifiers,
                     sensitive,
                     lca_funcs=lca_funcs,
                     range_funcs=range_funcs
                     )
    n = input_dataset.shape[0]
    mdl.suppress()
    output = mdl.dataset
    for i in range(n):
        assert np.all(output[i][field] == expected_output_dataset[i][field] for field in input_dataset.dtype.names)
        
input_concat_0 = np.array([('Brandon Liu', 'adamsdiane@gmail.com', 91, 16, 'male', '0123', 'hip', '21/01/2009'),
       ('Lori Erickson', 'millerclarence@gmail.com', 27, 62, 'male', '2312', 'heart', '10/01/2000'),
       ('Carolyn Adams', 'sethellis@smith.com', 78, 53, 'male', '3033', 'cancer', '18/12/2006')],
        dtype=[('Name', '<U16'), ('Email', '<U25'), ('Age', '<i4'), 
             ('Weight', '<i4'), ('Gender', '<U6'), ('Zipcode', '<U6'), 
             ('Diagnosis', '<U6'), ('Dob', '<U10')])

output_concat_0 = np.array([('Brandon Liu', 'adamsdiane@gmail.com', 91, 16, '0123', '21/01/2009', 'male-hip'),
       ('Lori Erickson', 'millerclarence@gmail.com', 27, 62, '2312', '10/01/2000', 'male-heart'),
       ('Carolyn Adams', 'sethellis@smith.com', 78, 53, '3033', '18/12/2006', 'male-cancer')],
        dtype=[('Name', '<U16'), ('Email', '<U25'), ('Age', '<i4'), 
             ('Weight', '<i4'), ('Zipcode', '<U6'), 
             ('Dob', '<U10'),('Diagnosis-Gender', '<U6')])

quasi_identifiers = ['Age', 'Weight', 'Gender', 'Zipcode', 'Dob']
sensitive = ['Diagnosis']

@pytest.mark.parametrize("input_dataset, identifiers, quasi_identifiers, sensitive, expected_output_dataset",
                         [(input_concat_0, identifiers, quasi_identifiers, sensitive, output_concat_0)])       
def test_concatenate_sensitive_attributes(input_dataset, identifiers, quasi_identifiers, sensitive, expected_output_dataset):
    mdl = BaseAnonymiser(input_dataset, 
                     identifiers,
                     quasi_identifiers,
                     sensitive,
                     lca_funcs=lca_funcs,
                     range_funcs=range_funcs
                     )
    n = input_dataset.shape[0]
    output = mdl.dataset
    for i in range(n):
        assert np.all(output[i][field] == expected_output_dataset[i][field] for field in input_dataset.dtype.names)

output_dtypes = [('Name', '<U16'), ('Email', '<U25'), ('Age', '<U9'), 
             ('Weight', '<U9'), ('Gender', '<U6'), ('Zipcode', '<U6'), 
             ('Diagnosis', '<U6'), ('Dob', '<U10')]

t0 = [item[0] for item in output_dtypes]
t1 = [item[1] for item in output_dtypes]

output_dtypes_0 = dict(zip(t0, t1))
@pytest.mark.parametrize("input_dataset, identifiers, quasi_identifiers, sensitive, expected_output_dtypes",
                         [(input_concat_0, identifiers, quasi_identifiers, sensitive, output_dtypes_0)])        
def test_change_numerical_to_str(input_dataset, identifiers, quasi_identifiers, sensitive, expected_output_dtypes):
    mdl = BaseAnonymiser(input_dataset, 
                     identifiers,
                     quasi_identifiers,
                     sensitive,
                     lca_funcs=lca_funcs,
                     range_funcs=range_funcs
                     )
    output_data = mdl.change_numerical_to_str(input_dataset)
    output = dict(output_data.dtype.fields)
    for key, val in expected_output_dtypes.items():
        assert key in output.keys()
        assert val == output[key][0]

input_ftr_range_0 = np.array([('Brandon Liu', 'adamsdiane@gmail.com', 91, 16, 'male', '0123', 'hip', '21/01/2009'),
       ('Lori Erickson', 'millerclarence@gmail.com', 27, 62, 'female', '2312', 'heart', '10/01/2000'),
       ('Carolyn Adams', 'sethellis@smith.com', 78, 53, 'male', '3033', 'cancer', '18/12/2006')],
        dtype=[('Name', '<U16'), ('Email', '<U25'), ('Age', '<i4'), 
             ('Weight', '<i4'), ('Gender', '<U6'), ('Zipcode', '<U6'), 
             ('Diagnosis', '<U6'), ('Dob', '<U10')])

expected_output_feature_ranges = {'Age': 64,
                                  'Weight': 46,
                                  'Gender': 1,
                                  'Zipcode': 4,
                                  'Dob': 5}

@pytest.mark.parametrize("input_dataset, identifiers, quasi_identifiers, sensitive, expected_output_feature_ranges",
                         [(input_ftr_range_0, identifiers, quasi_identifiers, sensitive, expected_output_feature_ranges)])        
def test_get_feature_ranges(input_dataset, identifiers, quasi_identifiers, sensitive, expected_output_feature_ranges):
    mdl = BaseAnonymiser(input_dataset, 
                     identifiers,
                     quasi_identifiers,
                     sensitive,
                     lca_funcs=lca_funcs,
                     range_funcs=range_funcs
                     )
    output_feature_ranges = mdl.get_feature_ranges(input_dataset)
    for key, val in expected_output_feature_ranges.items():
        assert key in output_feature_ranges
        assert np.all(val == output_feature_ranges[key])

input_filter_dataset = np.array([('Brandon Liu', 'adamsdiane@gmail.com', 91, 16, 'male', '0123', 'hip', '21/01/2009'),
       ('Lori Erickson', 'millerclarence@gmail.com', 27, 62, 'female', '2312', 'heart', '10/01/2000'),
       ('Carolyn Adams', 'sethellis@smith.com', 78, 53, 'male', '3033', 'cancer', '18/12/2006'),
       ('AA BB', 'aaaa@smith.com', 43, 87, 'female', '3223', 'cancer', '18/06/2006')],
        dtype=[('Name', '<U16'), ('Email', '<U25'), ('Age', '<i4'), 
             ('Weight', '<i4'), ('Gender', '<U6'), ('Zipcode', '<U6'), 
             ('Diagnosis', '<U6'), ('Dob', '<U10')])
  
output_filter_dataset = np.array([('Brandon Liu', 'adamsdiane@gmail.com', 91, 16, 'male', '0123', 'hip', '21/01/2009'),
       ('Lori Erickson', 'millerclarence@gmail.com', 27, 62, 'female', '2312', 'heart', '10/01/2000')],
        dtype=[('Name', '<U16'), ('Email', '<U25'), ('Age', '<i4'), 
             ('Weight', '<i4'), ('Gender', '<U6'), ('Zipcode', '<U6'), 
             ('Diagnosis', '<U6'), ('Dob', '<U10')])     
@pytest.mark.parametrize("input_dataset, identifiers, quasi_identifiers, sensitive, expected_output_filter_dataset",
                         [(input_filter_dataset, identifiers, quasi_identifiers, sensitive, output_filter_dataset)])        
def test_filter_dataset(input_dataset, identifiers, quasi_identifiers, sensitive, expected_output_filter_dataset):
    mdl = BaseAnonymiser(input_dataset, 
                     identifiers,
                     quasi_identifiers,
                     sensitive,
                     lca_funcs=lca_funcs,
                     range_funcs=range_funcs
                     )
    mdl.cluster_assignments = np.array([0, 0, 1, 1])
    output_filter_dataset = mdl.filter_dataset(input_dataset, [0])
    assert np.all(output_filter_dataset == expected_output_filter_dataset)