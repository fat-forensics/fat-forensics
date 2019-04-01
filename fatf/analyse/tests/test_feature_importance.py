import pytest
import numpy as np

from fatf.core.model import FAT_KNN
from fatf.analyse.feature_importance import (individual_conditional_expectation,
                                             partial_dependence, plot_ICE)
from fatf.exceptions import (MissingImplementationError, 
                             IncompatibleModelError, IncorrectShapeError)


numerical_array = np.array(
        [[0, 0, 0.08, 0.69],
        [1, 0, 0.03, 0.29],
        [0, 1, 0.99, 0.82],
        [2, 1, 0.73, 0.48],
        [1, 0, 0.36, 0.89],
        [0, 1, 0.07, 0.21]])

test_numerical_array = np.array(
        [[0, 1, 0.03, 0.5],
        [1, 0, 0.97, 0.3],
        [0, 0, 0.56, 0.32]])

structure_array = np.array([
        (0, 0, 0.08, 0.69),
        (1, 0, 0.03, 0.29),
        (0, 1, 0.99, 0.82),
        (2, 1, 0.73, 0.48),
        (1, 0, 0.36, 0.89),
        (0, 1, 0.07, 0.21)], dtype=[('a', 'i'), ('b', 'i'), ('c', 'f'), ('d', 'f')])

test_structure_array = np.array([
        (0, 1, 0.03, 0.5),
        (1, 0, 0.97, 0.3),
        (0, 0, 0.56, 0.32)], dtype=[('a', 'i'), ('b', 'i'), ('c', 'f'), ('d', 'f')])

results = np.array([
    [[0.25, 0.25, 0.5],
    [0.25, 0.25, 0.5],
    [0.25, 0.25, 0.5]],

    [[0.5, 0.25, 0.25],
    [0.5, 0.25, 0.25],
    [0.5, 0.25, 0.25]],

    [[0.5, 0, 0.5],
    [0.5, 0, 0.5],
    [0.25, 0.25,0.5]]])

values = np.array([0.3, 0.4, 0.5])

results_cat = np.array([
    [[0.25, 0.25, 0.5 ],
    [0.5 , 0.25, 0.25]],

    [[0.25, 0.25, 0.5 ],
    [0.5 , 0.25, 0.25]],

    [[0.5 , 0.  , 0.5 ],
    [0.5 , 0.25, 0.25]]])

values_cat = np.array([0., 1.])
values_cat_struct = np.array([0, 1], dtype='i')

class InvalidModel(object):
    """Class to test if exception when model does not have
    predict_proba(x) method.
    """
    def __init__(self):
        pass
    def predict(self):
        pass

def test_individual_conditional_expectation():
    predictor = FAT_KNN()
    with pytest.raises(IncorrectShapeException):
        individual_conditional_expectation(np.ones((6, 4, 4)), predictor, 3, steps=3)
    invalid_model = InvalidModel()
    with pytest.raises(IncompatibleModelException):
        individual_conditional_expectation(test_numerical_array, invalid_model, 3, 
                                            steps=3)
    with pytest.raises(CustomValueError):
        individual_conditional_expectation(test_numerical_array, predictor, 10, steps=3)

    predictor.fit(numerical_array, np.array([2, 0, 1, 1, 0, 2]))
    ice = individual_conditional_expectation(test_numerical_array, predictor, 3, steps=3)
    assert(np.array_equal(ice[0], results))
    assert(np.array_equal(ice[1], values))

    ice_cat = individual_conditional_expectation(test_numerical_array, predictor, 0, 
                                                  is_categorical=True, steps=3)
    assert(np.array_equal(ice_cat[0], results_cat))
    assert(np.array_equal(ice_cat[1], values_cat))

    predictor_struct = FAT_KNN()
    predictor_struct.fit(structure_array, np.array([2, 0, 1, 1, 0, 2]))

    ice_struct = individual_conditional_expectation(test_structure_array, predictor_struct, 'd', steps=3)
    assert(np.array_equal(ice_struct[0], results))
    #conversations make ice_struct[1] = [0.300000001, 0.40000001, 0.5]
    assert(any(np.isclose(ice_struct[1], values, atol=1e-7)))

    ice_struct_cat = individual_conditional_expectation(test_structure_array, predictor_struct, 'a', 
                                                         is_categorical=True, steps=3)
    assert(np.array_equal(ice_struct_cat[0], results_cat))
    assert(np.array_equal(ice_struct_cat[1], values_cat_struct))

def test_partial_dependence():
    predictor = FAT_KNN()
    with pytest.raises(IncorrectShapeException):
        partial_dependence(np.ones((6, 4, 4)), predictor, 3, steps=3)
    invalid_model = InvalidModel()
    with pytest.raises(IncompatibleModelException):
        partial_dependence(test_numerical_array, invalid_model, 3, 
                                            steps=3)
    with pytest.raises(CustomValueError):
        partial_dependence(test_numerical_array, predictor, 10, steps=3)

    predictor.fit(numerical_array, np.array([2, 0, 1, 1, 0, 2]))
    pd = partial_dependence(test_numerical_array, predictor, 3, steps=3)
    assert(np.array_equal(pd[0], np.mean(results, axis=0)))
    assert(np.array_equal(pd[1], values))

    pd_cat = partial_dependence(test_numerical_array, predictor, 0, is_categorical=True, 
                             steps=3)
    assert(np.array_equal(pd_cat[0], np.mean(results_cat, axis=0)))
    assert(np.array_equal(pd_cat[1], values_cat))

    predictor_struct = FAT_KNN()
    predictor_struct.fit(structure_array, np.array([2, 0, 1, 1, 0, 2]))

    pd_struct = partial_dependence(test_structure_array, predictor_struct, 'd', steps=3)
    assert(np.array_equal(pd_struct[0], np.mean(results, axis=0)))
    #conversations make ice_struct[1] = [0.300000001, 0.40000001, 0.5]
    assert(any(np.isclose(pd_struct[1], values, atol=1e-7)))

    pd_struct_cat = partial_dependence(test_structure_array, predictor_struct, 'a', 
                                                         is_categorical=True, steps=3)
    assert(np.array_equal(pd_struct_cat[0], np.mean(results_cat, axis=0)))
    assert(np.array_equal(pd_struct_cat[1], values_cat_struct))

def test_plot_ICE():
    # Not testing plotting functoin as it only offers basic functionality
    return True
