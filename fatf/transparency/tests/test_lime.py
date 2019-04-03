import pytest
import numpy as np

from fatf.transparency.lime import Lime
from fatf.utils.models import KNN
from fatf.exceptions import IncompatibleModelError, IncorrectShapeError

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

class InvalidModel(object):
    """Class to test if exception when model does not have
    predict_proba(x) method.
    """
    def __init__(self):
        pass
    def predict(self):
        pass

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
                break
            array1 = np.array([x[1] for x in val])
            array2 = np.array([x[1] for x in val2])
            if not np.isclose(array1, array2, atol=1).all():
                equal = False
                break
        else:
            equal = False
            break
    return equal

def test_lime_exceptions():
    # Test Exceptions
    with pytest.raises(NotImplementedError):
        lime = Lime(np.ones((6, 4), dtype='U4'), predictor, class_names=class_names,
                    feature_names=feature_names, random_state=10)
    invalid_model = InvalidModel()
    with pytest.raises(IncompatibleModelError):
        lime = Lime(numerical_array, invalid_model, class_names=class_names,
                    feature_names=feature_names, random_state=10)
    with pytest.raises(IncorrectShapeError):
        lime = Lime(np.ones((6, 4, 4)), predictor, class_names=class_names,
                    feature_names=feature_names, random_state=10)
    with pytest.raises(ValueError):
        lime = Lime(numerical_array, predictor, class_names=class_names,
                    feature_names=feature_names, random_state=10,
                    categorical_indices=np.array([0, 10]))
    with pytest.raises(ValueError):
        lime = Lime(numerical_array, predictor, class_names=['class1'],
                    feature_names=feature_names, random_state=10)
    with pytest.raises(ValueError):
        lime = Lime(numerical_array, predictor, class_names=class_names,
                    feature_names=['feature1'], random_state=10)

def test_explain_instance():
    sample = np.array([0, 1, 0.08, 0.54])
    lime = Lime(numerical_array, predictor, class_names=class_names, 
                feature_names=feature_names, random_state=10)
    explained = lime.explain_instance(sample)
    numerical_results = {
        'class0': [('feat0 <= 0.00', -0.5899970059619304), 
                   ('0.50 < feat1 <= 1.00', -0.23841203957061127), 
                   ('0.07 < feat2 <= 0.22', 0.08898846511033034), 
                   ('0.34 < feat3 <= 0.58', -0.0006522086523481255)], 
        'class1': [('0.50 < feat1 <= 1.00', 0.40233087463634754), 
                   ('0.07 < feat2 <= 0.22', -0.24281994305862944), 
                   ('feat0 <= 0.00', -0.14098229341952123), 
                   ('0.34 < feat3 <= 0.58', 0.026894158727456256)], 
        'class2': [('feat0 <= 0.00', 0.7309792993814516), 
                   ('0.50 < feat1 <= 1.00', -0.16391883506573632), 
                   ('0.07 < feat2 <= 0.22', 0.1538314779482988), 
                   ('0.34 < feat3 <= 0.58', -0.026241950075108125)]}
    assert(is_explain_equal(explained, numerical_results) == True)

    # change so feat0 and feat1 are treated as categorical
    lime = Lime(numerical_array, predictor, class_names=class_names, 
                feature_names=feature_names, random_state=10,
                categorical_indices=np.array([0, 1]))
    explained_categorical = lime.explain_instance(sample)
    categorical_results = {
        'class0': [('feat0=0', -0.5917320265412613), 
                   ('feat1=1', -0.2363129528114337), 
                   ('0.07 < feat2 <= 0.22', 0.09479678816923091), 
                   ('0.34 < feat3 <= 0.58', -0.0016313343281927754)], 
        'class1': [('feat1=1', 0.40432905210391235), 
                   ('0.07 < feat2 <= 0.22', -0.24457616053790646), 
                   ('feat0=0', -0.1405234157817066), 
                   ('0.34 < feat3 <= 0.58', 0.02509358329872823)], 
        'class2': [('feat0=0', 0.7322554423229684), 
                   ('feat1=1', -0.1680160992924789), 
                   ('0.07 < feat2 <= 0.22', 0.14977937236867547), 
                   ('0.34 < feat3 <= 0.58', -0.023462248970535445)]}
    assert(is_explain_equal(categorical_results,explained_categorical) == True)

    sample = np.array([(0, 1, 0.08, 0.54)], dtype=[('a', 'i'), ('b', 'i'), ('c', 'f'), ('d', 'f')])
    structure_array = np.array([
        (0, 0, 0.08, 0.69),
        (1, 0, 0.03, 0.29),
        (0, 1, 0.99, 0.82),
        (2, 1, 0.73, 0.48),
        (1, 0, 0.36, 0.89),
        (0, 1, 0.07, 0.21)], dtype=[('a', 'i'), ('b', 'i'), ('c', 'f'), ('d', 'f')])
    lime = Lime(structure_array, predictor, class_names=class_names,
                feature_names=feature_names, random_state=10)
    explained_numerical_structure = lime.explain_instance(sample)
    assert(is_explain_equal(explained_numerical_structure, numerical_results) == True)
    lime = Lime(structure_array, predictor, class_names=class_names,
                feature_names=feature_names, random_state=10,
                categorical_indices=np.array(['a', 'b']))
    explained_categorical_structure = lime.explain_instance(sample)
    assert(is_explain_equal(explained_categorical_structure, categorical_results) == True)
