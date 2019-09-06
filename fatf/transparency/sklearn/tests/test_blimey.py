"""
Tests Blimey class.
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
# License: new BSD

from typing import Dict, Union

import pytest

import numpy as np

from sklearn.linear_model import Ridge

import fatf

import fatf.transparency.sklearn.blimey as ftsmb
import fatf.utils.models as fum
import fatf.utils.data.augmentation as fuda
import fatf.utils.data.discretisation as fudd
import fatf.utils.transparency.explainers as fute
import fatf.utils.distances as fud
import fatf.utils.kernels as fuk
import fatf.transparency.sklearn.linear_model as ftslm

from fatf.exceptions import IncompatibleModelError, IncorrectShapeError

Index = Union[int, str]

ONE_D_ARRAY = np.array([0, 4, 3, 0])
NUMERICAL_NP_ARRAY_TARGET = np.array([2, 0, 1, 1, 0, 2])
NUMERICAL_NP_ARRAY = np.array([[0, 0, 0.08, 0.69], [1, 0, 0.03, 0.29],
                               [0, 1, 0.99, 0.82], [2, 1, 0.73, 0.48],
                               [1, 0, 0.36, 0.89], [0, 1, 0.07, 0.21]])
NUMERICAL_STRUCT_ARRAY = np.array([(0, 0, 0.08, 0.69), (1, 0, 0.03, 0.29),
                                   (0, 1, 0.99, 0.82), (2, 1, 0.73, 0.48),
                                   (1, 0, 0.36, 0.89), (0, 1, 0.07, 0.21)],
                                  dtype=[('a', 'i'), ('b', 'i'), ('c', 'f'),
                                         ('d', 'f')])
CATEGORICAL_NP_ARRAY = np.array([['a', 'b', 'c'], ['a', 'f', 'g'],
                                 ['b', 'c', 'c'], ['b', 'f', 'c'],
                                 ['a', 'f', 'c'], ['a', 'b', 'g']])
CATEGORICAL_STRUCT_ARRAY = np.array([('a', 'b', 'c'), ('a', 'f', 'g'),
                                     ('b', 'c', 'c'), ('b', 'f', 'c'),
                                     ('a', 'f', 'c'), ('a', 'b', 'g')],
                                    dtype=[('a', 'U1'), ('b', 'U1'), ('c',
                                                                      'U1')])
MIXED_ARRAY = np.array([(0, 'a', 0.08, 'a'), (0, 'f', 0.03, 'bb'),
                        (1, 'c', 0.99, 'aa'), (1, 'a', 0.73, 'a'),
                        (0, 'c', 0.36, 'b'), (1, 'f', 0.07, 'bb')],
                       dtype=[('a', 'i'), ('b', 'U1'), ('c', 'f'), ('d',
                                                                    'U2')])

NUMERICAL_NP_BLIMEY = {
    'Class A': {
        '0.07 < *C* <= 0.22': -0.0152,
        '0.58 < *D* <= 0.79': 0.0821,
        '*A* <= 0.00': -0.3665,
        '*B* <= 0.00': 0.1791,
    },
    'Class B': {
        '0.07 < *C* <= 0.22': 0.0546,
        '0.58 < *D* <= 0.79': 0.0073,
        '*A* <= 0.00': 0.1489,
        '*B* <= 0.00': -0.0791
    },
    'Class C': {
        '0.07 < *C* <= 0.22': -0.0393,
        '0.58 < *D* <= 0.79': -0.0894,
        '*A* <= 0.00': 0.2176,
        '*B* <= 0.00': -0.1000
    }
}

NUMERICAL_NP_BLIMEY_2 = {
    'Class A': {
        '*A* <= 0.00': -0.2503,
        '*B* <= 0.00': 0.1418
    },
    'Class B': {
        '*A* <= 0.00': 0.0724,
        '*B* <= 0.00': -0.1089
    },
    'Class C': {
        '*A* <= 0.00': 0.1778,
        '*B* <= 0.00': -0.0329
    }
}

NUMERICAL_NP_BLIMEY_CAT = {
    'Class A': {
        '0.07 < *C* <= 0.22': 0.0099,
        '0.58 < *D* <= 0.79': -0.0088,
        'A = 0': -0.1830,
        '*B* <= 0.00': 0.1897,
    },
    'Class B': {
        '0.07 < *C* <= 0.22': -0.0590,
            '0.58 < *D* <= 0.79': 0.0213,
            'A = 0': -0.0512,
            '*B* <= 0.00': -0.0354,
    },
    'Class C': {
        '0.07 < *C* <= 0.22': 0.0492,
        '0.58 < *D* <= 0.79': -0.0124,
        'A = 0': 0.2342,
        '*B* <= 0.00': -0.1543,
    }
}

NUMERICAL_NP_BLIMEY_NO_DISC = {
    'Class A': {
        'A': 0.2277,
        'B': -0.1952,
        'D': 0.0974
    },
    'Class B': {
        'A': -0.0807,
        'B': 0.0660,
        'C': 0.0830
    },
    'Class C': {
        'A': -0.1461,
        'B': 0.1342,
        'D': -0.1008
    }
}

NUMERICAL_STRUCT_BLIMEY = {
    'Class A': {
        'A = 0': -0.4191,
        'B = 0': 0.1507,
    },
    'Class B': {
        '0.58 < *D* <= 0.79': -0.1047,
        'B = 0': -0.1476,
    },
    'Class C': {
        '0.58 < *D* <= 0.79': 0.0482,
        'A = 0': 0.4240,
    }
}


class InvalidModel(object):
    """
    Tests for exceptions when a model lacks the ``predict_proba`` method.
    """

    def __init__(self):
        """
        Initialises not-a-model.
        """
        pass

    def fit(self, X, y):
        """
        Fits not-a-model.
        """
        return X, y  # pragma: nocover


def _is_explanation_equal(dict1: Dict[str, Dict[Index, np.float64]],
                          dict2: Dict[str, Dict[Index, np.float64]],
                          tol: float = 1e-2) -> bool:
    """
    Tests if two explanations are equal within a tolerance.
    """
    equal = True

    if set(dict1.keys()) == set(dict2.keys()):
        for key in dict1.keys():
            class_exp1 = dict1[key]
            class_exp2 = dict2[key]
            if set(class_exp1.keys()) == set(class_exp2.keys()):
                for feature in class_exp1.keys():
                    feat_vals1 = class_exp1[feature]
                    feat_vals2 = class_exp2[feature]
                    if not np.isclose(feat_vals1, feat_vals2, tol):
                        equal = False
            else:
                equal = False
    else:
        equal = False
    return equal


def test_input_is_valid():
    """
    Tests :func:`fatf.transparency.sklearn.blimey._is_input_valid`.
    """
    msg = 'The input dataset must be a 2-dimensional array.'
    with pytest.raises(IncorrectShapeError) as exin:
        ftsmb._input_is_valid(ONE_D_ARRAY, None, None, None, None, None)
    assert str(exin.value) == msg

    msg = ('The input dataset must only contain base types (textual and '
           'numerical).')
    with pytest.raises(TypeError) as exin:
        ftsmb._input_is_valid(
            np.array([[0, None], [0, 8]]), None, None, None, None, None)
    assert str(exin.value) == msg

    msg = ('This functionality requires the global model to be capable of '
           'outputting probabilities via predict_proba method.')
    model = InvalidModel()
    with pytest.raises(IncompatibleModelError) as exin:
        ftsmb._input_is_valid(
            NUMERICAL_NP_ARRAY,
            fuda.NormalSampling,
            ftslm.SKLearnLinearModelExplainer,
            model,
            None,
            discretiser=fudd.QuartileDiscretiser)
    assert str(exin.value) == msg

    msg = ('This functionality requires the local model to be capable of '
           'outputting predictions via predict method.')
    model = fum.KNN()
    with pytest.raises(IncompatibleModelError) as exin:
        ftsmb._input_is_valid(
            NUMERICAL_NP_ARRAY,
            fuda.NormalSampling,
            ftslm.SKLearnLinearModelExplainer,
            model,
            InvalidModel,
            discretiser=fudd.QuartileDiscretiser)
    assert str(exin.value) == msg

    msg = ('The following indices are invalid for the input dataset: '
           '{}.'.format(np.array(['a'])))
    with pytest.raises(IndexError) as exin:
        ftsmb._input_is_valid(
            NUMERICAL_NP_ARRAY,
            fuda.NormalSampling,
            ftslm.SKLearnLinearModelExplainer,
            model,
            fum.KNN, ['a'],
            discretiser=fudd.QuartileDiscretiser)
    assert str(exin.value) == msg

    msg = ('The categorical_indices parameter must be a Python list or None.')
    with pytest.raises(TypeError) as exin:
        ftsmb._input_is_valid(NUMERICAL_NP_ARRAY, fuda.NormalSampling,
                              ftslm.SKLearnLinearModelExplainer, model,
                              fum.KNN, 'a')
    assert str(exin.value) == msg

    msg = ('The augmentor object must inherit from abstract class '
           'fatf.utils.augmentation.Augmentation.')
    with pytest.raises(TypeError) as exin:
        ftsmb._input_is_valid(NUMERICAL_NP_ARRAY, InvalidModel,
                              ftslm.SKLearnLinearModelExplainer, model,
                              fum.KNN)
    assert str(exin.value) == msg

    msg = ('The discretiser object must be None or inherit from abstract '
           'class fatf.utils.discretisation.Discretiser.')
    with pytest.raises(TypeError) as exin:
        ftsmb._input_is_valid(
            NUMERICAL_NP_ARRAY,
            fuda.NormalSampling,
            ftslm.SKLearnLinearModelExplainer,
            model,
            fum.KNN,
            discretiser=InvalidModel)
    assert str(exin.value) == msg

    msg = ('The explainer object must inherit from abstract class fatf.utils.'
           'transparency.explainers.Explainer.')
    with pytest.raises(TypeError) as exin:
        ftsmb._input_is_valid(
            NUMERICAL_NP_ARRAY,
            fuda.NormalSampling,
            InvalidModel,
            model,
            fum.KNN,
            discretiser=fudd.QuartileDiscretiser)
    assert str(exin.value) == msg

    msg = 'The class_names parameter must be None or a list.'
    with pytest.raises(TypeError) as exin:
        ftsmb._input_is_valid(
            NUMERICAL_NP_ARRAY,
            fuda.NormalSampling,
            ftslm.SKLearnLinearModelExplainer,
            model,
            fum.KNN,
            class_names='a')
    assert str(exin.value) == msg

    msg = ('The class_name has to be either None or a string or a list of '
           'strings.')
    with pytest.raises(TypeError) as exin:
        ftsmb._input_is_valid(
            NUMERICAL_NP_ARRAY,
            fuda.NormalSampling,
            ftslm.SKLearnLinearModelExplainer,
            model,
            fum.KNN,
            class_names=[0])
    assert str(exin.value) == msg

    msg = 'The feature_names parameter must be None or a list.'
    with pytest.raises(TypeError) as exin:
        ftsmb._input_is_valid(
            NUMERICAL_NP_ARRAY,
            fuda.NormalSampling,
            ftslm.SKLearnLinearModelExplainer,
            model,
            fum.KNN,
            feature_names='a')
    assert str(exin.value) == msg

    msg = ('The length of feature_names must be equal to the number of '
           'features in the dataset.')
    with pytest.raises(ValueError) as exin:
        ftsmb._input_is_valid(
            NUMERICAL_NP_ARRAY,
            fuda.NormalSampling,
            ftslm.SKLearnLinearModelExplainer,
            model,
            fum.KNN,
            feature_names=['a'])
    assert str(exin.value) == msg

    msg = ('The feature name has to be either None or a string or a list of '
           'strings.')
    with pytest.raises(TypeError) as exin:
        ftsmb._input_is_valid(
            NUMERICAL_NP_ARRAY,
            fuda.NormalSampling,
            ftslm.SKLearnLinearModelExplainer,
            model,
            fum.KNN,
            feature_names=[0, 1, 2, 3])
    assert str(exin.value) == msg

    msg = ('discretise_first must be None or a boolean.')
    with pytest.raises(TypeError) as exin:
        ftsmb._input_is_valid(
            NUMERICAL_NP_ARRAY,
            fuda.NormalSampling,
            ftslm.SKLearnLinearModelExplainer,
            model,
            fum.KNN,
            discretise_first='a',
            discretiser=fudd.QuartileDiscretiser)
    assert str(exin.value) == msg

    msg = ('discretise_first is True but discretiser object is None. In order '
           'to discretise the sampled data prior to training a local model, '
           'please specify a discretiser object.')
    with pytest.raises(ValueError) as exin:
        ftsmb._input_is_valid(
            NUMERICAL_NP_ARRAY,
            fuda.NormalSampling,
            ftslm.SKLearnLinearModelExplainer,
            model,
            fum.KNN,
            discretise_first=True)
    assert str(exin.value) == msg

    msg = ('discretise_first is False but discretiser has been specified. The '
           'discretiser will be ignored and the data will not be discretised '
           'prior to training a local model.')
    with pytest.warns(UserWarning) as warning:
        ftsmb._input_is_valid(
            NUMERICAL_NP_ARRAY,
            fuda.NormalSampling,
            ftslm.SKLearnLinearModelExplainer,
            model,
            fum.KNN,
            discretise_first=False,
            discretiser=fudd.QuartileDiscretiser)
    assert len(warning) == 1
    assert str(warning[0].message) == msg

    # All ok
    assert ftsmb._input_is_valid(
        NUMERICAL_NP_ARRAY,
        fuda.NormalSampling,
        ftslm.SKLearnLinearModelExplainer,
        model,
        fum.KNN,
        discretiser=fudd.QuartileDiscretiser)

    assert ftsmb._input_is_valid(
        NUMERICAL_NP_ARRAY,
        fuda.NormalSampling,
        ftslm.SKLearnLinearModelExplainer,
        model,
        fum.KNN,
        discretiser=None)

    assert ftsmb._input_is_valid(
        NUMERICAL_NP_ARRAY,
        fuda.NormalSampling,
        ftslm.SKLearnLinearModelExplainer,
        model,
        fum.KNN,
        discretise_first=True,
        discretiser=fudd.QuartileDiscretiser)


class TestBlimey():
    """
    Tests :class:`fatf.transparency.sklearn.blimey.Blimey`.
    """
    knn_numerical = fum.KNN(k=3)
    knn_numerical.fit(NUMERICAL_NP_ARRAY, NUMERICAL_NP_ARRAY_TARGET)

    knn_numerical_structured = fum.KNN(k=3)
    knn_numerical_structured.fit(NUMERICAL_STRUCT_ARRAY,
                                 NUMERICAL_NP_ARRAY_TARGET)

    knn_categorical = fum.KNN(k=3)
    knn_categorical.fit(CATEGORICAL_NP_ARRAY, NUMERICAL_NP_ARRAY_TARGET)

    knn_categorical_structured = fum.KNN(k=3)
    knn_categorical_structured.fit(CATEGORICAL_STRUCT_ARRAY,
                                   NUMERICAL_NP_ARRAY_TARGET)

    knn_mixed = fum.KNN(k=3)
    knn_mixed.fit(MIXED_ARRAY, NUMERICAL_NP_ARRAY_TARGET)

    class_names = ['Class A', 'Class B', 'Class C']
    feature_names = ['A', 'B', 'C', 'D']

    numerical_blimey = ftsmb.Blimey(
        dataset=NUMERICAL_NP_ARRAY,
        augmentor=fuda.NormalSampling,
        explainer=ftslm.SKLearnLinearModelExplainer,
        global_model=knn_numerical,
        local_model=Ridge,
        class_names=class_names,
        feature_names=feature_names,
        discretiser=fudd.QuartileDiscretiser)

    numerical_blimey_cat = ftsmb.Blimey(
        dataset=NUMERICAL_NP_ARRAY,
        augmentor=fuda.NormalSampling,
        explainer=ftslm.SKLearnLinearModelExplainer,
        global_model=knn_numerical,
        local_model=Ridge,
        class_names=class_names,
        feature_names=feature_names,
        categorical_indices=[0],
        discretiser=fudd.QuartileDiscretiser)

    numerical_blimey_no_discretisation = ftsmb.Blimey(
        dataset=NUMERICAL_NP_ARRAY,
        augmentor=fuda.NormalSampling,
        explainer=ftslm.SKLearnLinearModelExplainer,
        global_model=knn_numerical,
        local_model=Ridge,
        class_names=class_names,
        feature_names=feature_names,
        discretise_first=False)

    numerical_blimey_structured = ftsmb.Blimey(
        dataset=NUMERICAL_STRUCT_ARRAY,
        augmentor=fuda.NormalSampling,
        explainer=ftslm.SKLearnLinearModelExplainer,
        global_model=knn_numerical_structured,
        local_model=Ridge,
        class_names=class_names,
        feature_names=feature_names,
        discretiser=fudd.QuartileDiscretiser,
        categorical_indices=['a', 'b'])

    categorical_blimey = ftsmb.Blimey(
        dataset=CATEGORICAL_NP_ARRAY,
        augmentor=fuda.NormalSampling,
        explainer=ftslm.SKLearnLinearModelExplainer,
        global_model=knn_categorical,
        local_model=Ridge,
        discretise_first=False)

    msg = ('Some of the string-based columns in the input dataset '
           'were not selected as categorical features via the '
           'categorical_indices parameter. String-based columns '
           'cannot be treated as numerical features, therefore '
           'they will be also treated as categorical features '
           '(in addition to the ones selected with the '
           'categorical_indices parameter).')
    with pytest.warns(UserWarning) as warning:
        categorical_blimey_structured = ftsmb.Blimey(
            dataset=CATEGORICAL_STRUCT_ARRAY,
            augmentor=fuda.NormalSampling,
            explainer=ftslm.SKLearnLinearModelExplainer,
            global_model=knn_categorical_structured,
            local_model=Ridge,
            discretise_first=False,
            categorical_indices=['a'])
    assert len(warning) == 1
    assert str(warning[0].message) == msg

    mixed_blimey = ftsmb.Blimey(
        dataset=MIXED_ARRAY,
        augmentor=fuda.NormalSampling,
        explainer=ftslm.SKLearnLinearModelExplainer,
        global_model=knn_mixed,
        local_model=Ridge,
        class_names=class_names,
        feature_names=feature_names,
        discretiser=fudd.QuartileDiscretiser)

    def test_init(self):
        """
        Tests :class:`fatf.transparency.sklearn.blimey.Blimey` class init.
        """

        # Class inherits from explainer but is not tree or linear
        class ExplainerUnknown(fute.Explainer):
            """
            Explainer that is not an sklearn explainer.
            """

            def __init__(self):
                pass

            def explain_model(self):
                pass

            def explain_instance(self):
                pass

        # Infer discretise_first
        blimey_infer_discretise = ftsmb.Blimey(
            dataset=NUMERICAL_NP_ARRAY,
            augmentor=fuda.NormalSampling,
            explainer=ftslm.SKLearnLinearModelExplainer,
            global_model=self.knn_numerical,
            local_model=Ridge,
            categorical_indices=[0],
            discretiser=fudd.QuartileDiscretiser)

        assert blimey_infer_discretise.discretise_first
        '''
        # With SKLearnDecisionTreeExplainer
        blimey_infer_discretise = ftsmb.Blimey(
            dataset=NUMERICAL_NP_ARRAY,
            augmentor=fuda.NormalSampling,
            explainer=ftslm.SKLearnDecisionTreeExplainer,
            global_model=knn_numerical,
            local_model=Ridge,
            class_names=['Class A', 'Class B', 'Class C'],
            feature_names=['A', 'B', 'C', 'D'],
            categorical_indices=[0])

        assert blimey_infer_discretise.discretise_first is False
        '''

        msg = ('Unable to infer value of discretise_first as the explainer '
               'used is not one of the explainers defined in '
               'fatf.transparency.sklearn')
        with pytest.raises(ValueError) as exin:
            blimey_infer_discretise = ftsmb.Blimey(
                dataset=NUMERICAL_NP_ARRAY,
                augmentor=fuda.NormalSampling,
                explainer=ExplainerUnknown,
                global_model=self.knn_numerical,
                local_model=Ridge,
                categorical_indices=[0],
                discretiser=fudd.QuartileDiscretiser)
        assert str(exin.value) == msg

        msg = ('discretise_first has been inferred to be True given the type '
               'of explainer, but a discretiser object has not been given. As '
               'such, the data will not be discretised.')
        with pytest.warns(UserWarning) as warning:
            blimey_infer_discretise = ftsmb.Blimey(
                dataset=NUMERICAL_NP_ARRAY,
                augmentor=fuda.NormalSampling,
                explainer=ftslm.SKLearnLinearModelExplainer,
                global_model=self.knn_numerical,
                local_model=Ridge,
                categorical_indices=[0])
        assert len(warning) == 1
        assert str(warning[0].message) == msg
        assert blimey_infer_discretise.discretise_first is False

        # numerical_blimey
        assert self.numerical_blimey.discretise_first is True
        assert self.numerical_blimey.is_structured is False
        assert self.numerical_blimey.categorical_indices == []
        assert self.numerical_blimey.numerical_indices == [0, 1, 2, 3]
        assert self.numerical_blimey.class_names == self.class_names
        assert self.numerical_blimey.feature_names == self.feature_names

        # numerical_blimey_cat
        assert self.numerical_blimey_cat.discretise_first is True
        assert self.numerical_blimey_cat.is_structured is False
        assert self.numerical_blimey_cat.categorical_indices == [0]
        assert self.numerical_blimey_cat.numerical_indices == [1, 2, 3]
        assert self.numerical_blimey_cat.class_names == self.class_names
        assert self.numerical_blimey_cat.feature_names == self.feature_names

        # numerical_blimey_no_discretisation
        assert self.numerical_blimey_no_discretisation.discretise_first \
            is False
        assert self.numerical_blimey_no_discretisation.is_structured is False
        assert self.numerical_blimey_no_discretisation.categorical_indices \
            == []
        assert self.numerical_blimey_no_discretisation.numerical_indices == [
            0, 1, 2, 3
        ]
        assert self.numerical_blimey_cat.class_names == self.class_names
        assert self.numerical_blimey_no_discretisation.feature_names \
            == self.feature_names

        # numerical_blimey_structured
        assert self.numerical_blimey_structured.discretise_first is True
        assert self.numerical_blimey_structured.is_structured is True
        assert self.numerical_blimey_structured.categorical_indices == [
            'a', 'b'
        ]
        assert self.numerical_blimey_structured.numerical_indices == ['c', 'd']
        assert self.numerical_blimey_structured.class_names == self.class_names
        assert self.numerical_blimey_structured.feature_names == \
            self.feature_names

        # categorical_blimey
        assert self.categorical_blimey.discretise_first is False
        assert self.categorical_blimey.is_structured is False
        assert self.categorical_blimey.categorical_indices == [0, 1, 2]
        assert self.categorical_blimey.numerical_indices == []
        assert self.categorical_blimey.class_names == [
            'class 0', 'class 1', 'class 2'
        ]
        assert self.categorical_blimey.feature_names == [
            'feature 0', 'feature 1', 'feature 2'
        ]

        # categorical_blimey_structured
        assert self.categorical_blimey_structured.discretise_first is False
        assert self.categorical_blimey_structured.is_structured is True
        assert self.categorical_blimey_structured.categorical_indices == [
            'a', 'b', 'c'
        ]
        assert self.categorical_blimey_structured.numerical_indices == []
        assert self.categorical_blimey_structured.class_names == [
            'class 0', 'class 1', 'class 2'
        ]
        assert self.categorical_blimey.feature_names == [
            'feature 0', 'feature 1', 'feature 2'
        ]

        # mixed_blimey
        assert self.mixed_blimey.discretise_first is True
        assert self.mixed_blimey.is_structured is True
        assert self.mixed_blimey.categorical_indices == ['b', 'd']
        assert self.mixed_blimey.numerical_indices == ['a', 'c']
        assert self.mixed_blimey.class_names == self.class_names
        assert self.mixed_blimey.feature_names == self.feature_names

    def test_explain_instance_is_input_valid(self):
        """
        Tests :func:`fatf.transparency.sklearn.blimey.Blimey._explain_instance
        _is_input_valid`.
        """
        shape_err_msg = ('data_row must be a 1-dimensional array')
        dtype_err = ('The dtype of the data is different to '
                     'the dtype of the data array used to '
                     'initialise this class.')
        features_shape_msg = ('The data must contain the same number of '
                              'features as the dataset used to initialise '
                              'this class.')
        samples_number_value_msg = ('The samples_number parameter must be a '
                                    'positive integer.')
        samples_number_type_msg = ('The samples_number parameter must be an '
                                   'integer.')
        kernel_function_msg = ('The kernel function must have only 1 required '
                               'parameter. Any additional parameters can be '
                               'passed via **kwargs.')
        distance_function_msg = ('The distance function must have only 2 '
                                 'required parameters. Any additional '
                                 'parameters can be passed via **kwargs.')
        features_number_value_msg = ('The features_number parameter must be a '
                                     'positive integer.')
        features_number_type_msg = ('The features_number parameter must be an '
                                    'integer.')
        features_number_warning = ('features_number is larger than the number '
                                   'of features in the dataset, therefore all '
                                   'features will be used.')

        def valid_kernel(X):
            pass

        def invalid_kernel(X=3, Y=3):
            pass

        def valid_distance(X, Y):
            pass

        def invalid_distance(X):
            pass

        with pytest.raises(IncorrectShapeError) as exin:
            self.numerical_blimey._explain_instance_is_input_valid(
                NUMERICAL_NP_ARRAY, 1, None, None, None)
        assert str(exin.value) == shape_err_msg

        with pytest.raises(IncorrectShapeError) as exin:
            self.categorical_blimey._explain_instance_is_input_valid(
                CATEGORICAL_NP_ARRAY, 1, None, None, None)
        assert str(exin.value) == shape_err_msg

        with pytest.raises(TypeError) as exin:
            self.numerical_blimey._explain_instance_is_input_valid(
                CATEGORICAL_NP_ARRAY[0], 1, None, None, None)
        assert str(exin.value) == dtype_err

        with pytest.raises(TypeError) as exin:
            self.numerical_blimey_structured._explain_instance_is_input_valid(
                CATEGORICAL_STRUCT_ARRAY[0], 1, None, None, None)
        assert str(exin.value) == dtype_err

        with pytest.raises(IncorrectShapeError) as exin:
            self.numerical_blimey._explain_instance_is_input_valid(
                NUMERICAL_NP_ARRAY[0][0:3], 1, None, None, None)
        assert str(exin.value) == features_shape_msg

        with pytest.raises(ValueError) as exin:
            self.numerical_blimey._explain_instance_is_input_valid(
                NUMERICAL_NP_ARRAY[0], -1, None, None, None)
        assert str(exin.value) == samples_number_value_msg

        with pytest.raises(TypeError) as exin:
            self.numerical_blimey._explain_instance_is_input_valid(
                NUMERICAL_NP_ARRAY[0], 'a', None, None, None)
        assert str(exin.value) == samples_number_type_msg

        with pytest.raises(TypeError) as exin:
            self.numerical_blimey._explain_instance_is_input_valid(
                NUMERICAL_NP_ARRAY[0], 1, invalid_kernel, None, None)
        assert str(exin.value) == kernel_function_msg

        with pytest.raises(TypeError) as exin:
            self.numerical_blimey._explain_instance_is_input_valid(
                NUMERICAL_NP_ARRAY[0], 1, valid_kernel, invalid_distance, None)
        assert str(exin.value) == distance_function_msg

        with pytest.raises(ValueError) as exin:
            self.numerical_blimey._explain_instance_is_input_valid(
                NUMERICAL_NP_ARRAY[0], 1, valid_kernel, valid_distance, -1)
        assert str(exin.value) == features_number_value_msg

        with pytest.raises(TypeError) as exin:
            self.numerical_blimey._explain_instance_is_input_valid(
                NUMERICAL_NP_ARRAY[0], 1, valid_kernel, valid_distance, 'a')
        assert str(exin.value) == features_number_type_msg

        with pytest.warns(UserWarning) as warning:
            self.numerical_blimey._explain_instance_is_input_valid(
                NUMERICAL_NP_ARRAY[0], 1, valid_kernel, valid_distance, 1000)
        assert len(warning) == 1
        assert str(warning[0].message) == features_number_warning

        # All good
        assert self.numerical_blimey._explain_instance_is_input_valid(
            NUMERICAL_NP_ARRAY[0], 10, valid_kernel, valid_distance, 2)
        assert self.categorical_blimey._explain_instance_is_input_valid(
            CATEGORICAL_NP_ARRAY[0], 10, valid_kernel, valid_distance, 2)
        assert (
            self.numerical_blimey_structured._explain_instance_is_input_valid(
                NUMERICAL_STRUCT_ARRAY[0], 10, valid_kernel, valid_distance,
                2))
        assert self.mixed_blimey._explain_instance_is_input_valid(
            MIXED_ARRAY[0], 10, valid_kernel, valid_distance, 2)

    def test_explain_instance(self):
        """
        Tests :func:`fatf.transparency.sklearn.blimey.Blimey.explain_instance`.
        """

        msg = ('The local_model is a sklearn model but the data has '
               'non-numerical values. Sklearn models are incompatible '
               'with non-numerical values, please either use a different '
               'local_model or one hot encode the non-numerical values.')
        features_number_warning = ('features_number is larger than the number '
                                   'of features in the dataset, therefore all '
                                   'features will be used.')

        with pytest.raises(IncompatibleModelError) as exin:
            exp = self.categorical_blimey.explain_instance(
                CATEGORICAL_NP_ARRAY[0])
        assert str(exin.value) == msg

        fatf.setup_random_seed()
        exp = self.numerical_blimey.explain_instance(
            data_row=NUMERICAL_NP_ARRAY[0],
            samples_number=50,
            kernel_function=None,
            distance_function=fud.euclidean_array_distance,
            features_number=None)
        assert _is_explanation_equal(exp, NUMERICAL_NP_BLIMEY)

        exp = self.numerical_blimey.explain_instance(
            data_row=NUMERICAL_NP_ARRAY[0],
            samples_number=50,
            kernel_function=fuk.exponential_kernel,
            width=0.9,
            distance_function=fud.euclidean_array_distance,
            features_number=2)
        assert _is_explanation_equal(exp, NUMERICAL_NP_BLIMEY_2)

        with pytest.warns(UserWarning) as warning:
            exp = self.numerical_blimey_cat.explain_instance(
                data_row=NUMERICAL_NP_ARRAY[0],
                samples_number=50,
                kernel_function=None,
                distance_function=fud.binary_array_distance,
                features_number=1000)
        assert len(warning) == 1
        assert str(warning[0].message) == features_number_warning
        assert _is_explanation_equal(exp, NUMERICAL_NP_BLIMEY_CAT)

        exp = self.numerical_blimey_no_discretisation.explain_instance(
            data_row=NUMERICAL_NP_ARRAY[0],
            samples_number=50,
            kernel_function=fuk.exponential_kernel,
            width=0.5,
            distance_function=fud.euclidean_array_distance,
            features_number=3)
        assert _is_explanation_equal(exp, NUMERICAL_NP_BLIMEY_NO_DISC)

        # No distances from instance used to train local model
        exp = self.numerical_blimey_structured.explain_instance(
            data_row=NUMERICAL_STRUCT_ARRAY[0],
            samples_number=50,
            kernel_function=None,
            features_number=2)
        assert _is_explanation_equal(exp, NUMERICAL_STRUCT_BLIMEY)
