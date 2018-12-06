# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 09:33:43 2018

@author: Rafael Poyiadzi
"""
import numpy as np
from collections import Counter

from fatf.utils.validation import check_array_type

testdata = np.array([('Heidi Mitchell', 'uboyd@hotmail.com', 74, 52, 'female', '1121', 'cancer', '03/06/2018'),
       ('Tina Burns', 'stevenwheeler@williams.bi',  3, 86, 'female', '0323', 'hip', '26/09/2017'),
       ('Justin Brown', 'velasquezjake@gmail.com', 26, 56, 'female', '0100', 'heart', '31/12/2015'),
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
      dtype=[('name', '<U16'), ('email', '<U25'), ('age', '<i4'), ('weight', '<i4'), ('gender', '<U6'), ('zipcode', '<U6'), ('diagnosis', '<U6'), ('dob', '<U10')])

testdata2 = np.array([('Heidi Mitchell', 'uboyd@hotmail.com', 74, 52, 'female', '1121', 'cancer', '03/06/2018'),
       ('Tina Burns', 'stevenwheeler@williams.bi',  3, 86, 'female', '0323', 'hip', '26/09/2017'),
       ('Justin Brown', 'velasquezjake@gmail.com', 26, 56, 'female', '0100', 'heart', '31/12/2015'),
       ('Brent Parker', 'kennethsingh@strong-foley', 70, 57, 'male', '3131', 'heart', '02/10/2011'),
       ('Bryan Norton', 'erica36@hotmail.com', 48, 57, 'male', '0301', 'hip', '09/09/2012'),
       ('Ms. Erin Craig', 'ritterluke@gmail.com', 30, 98, 'male', '2223', 'cancer', '04/11/2006'),],
      dtype=[('name', '<U16'), ('email', '<U25'), ('age', '<i4'), ('weight', '<i4'), ('gender', '<U6'), ('zipcode', '<U6'), ('diagnosis', '<U6'), ('dob', '<U10')])

def describe_numeric(series):
    numeric_dict = {}
    numeric_dict['count'] = int(len(series))
    numeric_dict['mean'] = np.mean(series)
    numeric_dict['std'] = np.std(series)
    numeric_dict['max'] = np.max(series)
    numeric_dict['min'] = np.min(series)
    numeric_dict['25%'] = np.quantile(series, 0.25)
    numeric_dict['50%'] = np.quantile(series, 0.50)
    numeric_dict['75%'] = np.quantile(series, 0.75)

    for key, value in numeric_dict.items():
        if key != 'count':
            numeric_dict[key] = format(value, '.2f')
    return numeric_dict

def describe_categorical(series):
    categorical_dict = {}
    categorical_dict['count'] = int(len(series))
    categorical_dict['count_unique'] = len(set(series))
    categorical_dict['unique'] = list(set(series))

    counter = Counter(series)
    top = counter.most_common()
    categorical_dict['most_common'] = top[0][0]
    categorical_dict['most_common_count'] = top[0][1]

    categorical_dict['hist'] = dict(counter)

    return categorical_dict

def describe_dataset(dataset, todescribe='all', condition=None):
    numerical_fields, categorical_fields = check_array_type(dataset)
    if todescribe == 'all':
        todescribe = numerical_fields.tolist() + categorical_fields.tolist()
    if condition is not None:
        values_set = list(set(condition))
        n_samples = condition.shape[0]
        grand_dict = {}
        for value in values_set:
            mask = np.array(np.zeros(n_samples), dtype=bool)
            t = np.where(condition == value)[0]
            mask[t] = True
            describe_dict = {}

            for field_name in numerical_fields:
                if field_name in todescribe:
                    describe_dict[field_name] = describe_numeric(dataset[mask][field_name])
            for field_name in categorical_fields:
                if field_name in todescribe:
                    describe_dict[field_name] = describe_categorical(dataset[mask][field_name])
            grand_dict[value] = describe_dict
        return grand_dict
    else:
        describe_dict = {}
        for field_name in numerical_fields:
            if field_name in todescribe:
                describe_dict[field_name] = describe_numeric(dataset[field_name])
        for field_name in categorical_fields:
            if field_name in todescribe:
                describe_dict[field_name] = describe_categorical(dataset[field_name])
        return describe_dict

condition = np.array(['m', 'm', 'm', 'f', 'f', 'f'])
a=describe_dataset(testdata2, ['diagnosis'])
