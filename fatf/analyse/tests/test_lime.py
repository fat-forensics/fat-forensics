import pytest
import numpy as np

from fatf.analyse.lime import Lime, plot_lime
from fatf.tests.predictor import KNN
from fatf.exceptions import *


def is_explain_equal(dict1, dict2):
    """Small functions to test if the two dictionarys are equal 
    to certain accuracy (due to float numbers). Assume dicts are 
    returned from Lime.explain_instance method.
    """
    equal = True
    for key, val in dict1.items():
        if key in dict2:
            val2 = dict2[key]
            if set([x[0] for x in val]) != set([x[0] for x in val2]):
                equal = False
            array1 = np.array([x[1] for x in val])
            array2 = np.array([x[1] for x in val2])
            # TODO: not sure how to test if still returning same numbers
            # apart from actually importing LIME package and using it
            # during testing
            if not any(np.isclose(array1, array2), atol=1e-1):
                equal = False
    return equal

def test_explain_instance():
    numerical_array = np.array(
        [[0, 0, 0.08, 0.69],
        [1, 0, 0.03, 0.29],
        [0, 1, 0.99, 0.82],
        [2, 1, 0.73, 0.48],
        [1, 0, 0.36, 0.89],
        [0, 1, 0.07, 0.21]])
    predictor = KNN()
    predictor.fit(numerical_array, np.array([2, 0, 1, 1, 0, 2]))
    class_names = ['class0', 'class1', 'class2']
    feature_names = ['feat0', 'feat1', 'feat2', 'feat3']

    # Test Exceptions
    with pytest.raises(MissingImplementationException):
        lime = Lime(np.ones((6, 4), dtype='U4'), predictor, class_names=class_names,
                    feature_names=feature_names)
    with pytest.raises(CustomValueError):
        lime = Lime(numerical_array, predictor, class_names=['class1'],
                    feature_names=feature_names)
    with pytest.raises(CustomValueError):
        lime = Lime(numerical_array, predictor, class_names=class_names,
                    feature_names=['feature1'])

    sample = np.array([0, 1, 0.08, 0.54])
    lime = Lime(numerical_array, predictor, class_names=class_names, 
                feature_names=feature_names)
    explained = lime.explain_instance(sample)
    numerical_results = {
        'class0': [('feat0 <= 0.00', -0.5837300458177844), 
                   ('0.50 < feat1 <= 1.00', -0.2555361894705736),
                   ('0.07 < feat2 <= 0.22', 0.08791870040934073), 
                   ('0.34 < feat3 <= 0.58', -0.01174869730270592)], 
        'class1': [('0.50 < feat1 <= 1.00', 0.4087292942591764), 
                   ('0.07 < feat2 <= 0.22', -0.22504492480948446), 
                   ('feat0 <= 0.00', -0.17409522917266232), 
                   ('0.34 < feat3 <= 0.58', 0.026424659734707926)], 
        'class2': [('feat0 <= 0.00', 0.7578252749904465), 
                   ('0.50 < feat1 <= 1.00', -0.1531931047886031), 
                   ('0.07 < feat2 <= 0.22', 0.13712622440014366), 
                   ('0.34 < feat3 <= 0.58', -0.014675962432001989)]
    }
    assert(is_explain_equal(numerical_results, explained) == True)

    # change so feat0 and feat1 are treated as categorical
    lime.categorical_indices = np.array([0, 1])
    explained_categorical = lime.explain_instance(sample)
    categorical_results = {
        'class0': [('feat0=0', -0.5792437412121086), 
                   ('feat1=1', -0.23454352789343055), 
                   ('0.07 < feat2 <= 0.22', 0.08503258506769569), 
                   ('0.34 < feat3 <= 0.58', -0.004202901435275888)], 
        'class1': [('feat1=1', 0.3973356302724271), 
                   ('0.07 < feat2 <= 0.22', -0.23094538303057252), 
                   ('feat0=0', -0.17855494136303698), 
                   ('0.34 < feat3 <= 0.58', 0.01744819782789471)], 
        'class2':[('feat0=0', 0.7577986825751453), 
                  ('feat1=1', -0.16279210237899622), 
                  ('0.07 < feat2 <= 0.22', 0.14591279796287623), 
                  ('0.34 < feat3 <= 0.58', -0.013245296392618885)]
    }
    assert(is_explain_equal(categorical_results,explained_categorical) == True)

def test_plot():
    #TODO: write plot tests - not sure what it has to test
    assert(0==0)
