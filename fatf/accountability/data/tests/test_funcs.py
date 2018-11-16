# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 14:43:54 2018

@author: Rafael Poyiadzi
"""

import pytest
import numpy as np

from funcs import generalize_string, suppress, concatenate_sensitive_attributes

@pytest.mark.parametrize("input_string, expected_string", 
                    [('1234', '123*'),
                     ('123*', '12**'),
                     ('****', '****')])
def test_generalize_string(input_string, expected_string):
    assert generalize_string(input_string) == expected_string
    
    
input0 = np.array([('Brandon Liu', 'adamsdiane@gmail.com', 91, 16, 'male', '0123', 'hip', '21/01/2009'),
       ('Lori Erickson', 'millerclarence@gmail.com', 27, 62, 'male', '2312', 'heart', '10/01/2000'),
       ('Carolyn Adams', 'sethellis@smith.com', 78, 53, 'male', '3033', 'cancer', '18/12/2006'),
       ('Donna Orr', 'jgibson@hunter.com', 50, 81, 'female', '1013', 'cancer', '14/01/2002'),
       ('Jennifer Cook', 'ksingleton@brown.com', 64, 10, 'male', '0000', 'cancer', '12/10/2004'),
       ('Kara Cunningham ', 'strongbrittany@gmail.com', 88,  6, 'female', '1013', 'lung', '28/03/2005'),],
      dtype=[('name', '<U16'), ('email', '<U25'), ('age', '<i4'), ('weight', '<i4'), ('gender', '<U6'), ('zipcode', '<U6'), ('diagnosis', '<U6'), ('dob', '<U10')])

output0 = np.array([('*', '*', 91, 16, 'male', '0123', 'hip', '21/01/2009'),
       ('*', '*', 27, 62, 'male', '2312', 'heart', '10/01/2000'),
       ('*', '*', 78, 53, 'male', '3033', 'cancer', '18/12/2006'),
       ('*', '*', 50, 81, 'female', '1013', 'cancer', '14/01/2002'),
       ('*', '*', 64, 10, 'male', '0000', 'cancer', '12/10/2004'),
       ('* ', '*', 88,  6, 'female', '1013', 'lung', '28/03/2005'),],
      dtype=[('name', '<U16'), ('email', '<U25'), ('age', '<i4'), ('weight', '<i4'), ('gender', '<U6'), ('zipcode', '<U6'), ('diagnosis', '<U6'), ('dob', '<U10')])


@pytest.mark.parametrize("input_dataset, expected_output_dataset, attributes_to_supress", 
                         [(input0, output0, ['name', 'email'])])
def test_suppress(input_dataset, expected_output_dataset, attributes_to_supress):
    n = input_dataset.shape[0]
    output = suppress(input_dataset, attributes_to_supress)
    for i in range(n):
        assert np.all(output[i][field] == expected_output_dataset[i][field] for field in input_dataset.dtype.names)
        
input_concat_0 = np.array([('Brandon Liu', 'adamsdiane@gmail.com', 91, 16, 'male', '0123', 'hip', '21/01/2009'),
       ('Lori Erickson', 'millerclarence@gmail.com', 27, 62, 'male', '2312', 'heart', '10/01/2000'),
       ('Carolyn Adams', 'sethellis@smith.com', 78, 53, 'male', '3033', 'cancer', '18/12/2006')],
      dtype=[('name', '<U16'), ('email', '<U25'), ('age', '<i4'), ('weight', '<i4'), ('gender', '<U6'), ('zipcode', '<U6'), ('diagnosis', '<U6'), ('dob', '<U10')])

output_concat_0 = np.array([('Brandon Liu', 'adamsdiane@gmail.com', 91, 16, '0123', '21/01/2009', 'male-hip'),
       ('Lori Erickson', 'millerclarence@gmail.com', 27, 62, '2312', '10/01/2000', 'male-heart'),
       ('Carolyn Adams', 'sethellis@smith.com', 78, 53, '3033', '18/12/2006', 'male-cancer')],
      dtype=[('name', '<U16'), ('email', '<U25'), ('age', '<i4'), ('weight', '<i4'), ('zipcode', '<U6'), ('dob', '<U10'), ('newfield', '<U30')])

@pytest.mark.parametrize("input_dataset, expected_output_dataset, sensitive_attributes",
                         [(input_concat_0, output_concat_0, ['diagnosis', 'gender'])])       
def test_concatenate_sensitive_attributes(input_dataset, expected_output_dataset, sensitive_attributes):
    n = input_dataset.shape[0]
    output = concatenate_sensitive_attributes(input_dataset, sensitive_attributes)
    for i in range(n):
        assert np.all(output[i][field] == expected_output_dataset[i][field] for field in input_dataset.dtype.names)