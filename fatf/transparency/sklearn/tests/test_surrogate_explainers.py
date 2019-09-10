"""
Tests surrogate_explainers classes.
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
# License: new BSD

from typing import Dict, Union

import pytest

import numpy as np

from sklearn.linear_model import Ridge

import fatf

import fatf.transparency.sklearn.surrogate_explainers as ftsse
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


class NonProbaModel(object):
    """
    Tests for exceptions when model has ``predict`` method.
    """

    def __init__(self):
        """
        Initialises non-probabilistic.
        """
        pass

    def fit(self, X, y):
        """
        Fits non-probabilistic.
        """
        return X, y

    def predict(self, x):
        """
        Predicts non-probabilistic model.
        """
        pass


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
    Tests :func:`fatf.transparency.sklearn.surrogate_explainers._is_input_valid`.
    """
    msg = 'The input dataset must be a 2-dimensional array.'
    with pytest.raises(IncorrectShapeError) as exin:
        ftsse._input_is_valid(ONE_D_ARRAY, None, None, None, None, None)
    assert str(exin.value) == msg

    msg = ('The input dataset must only contain base types (textual and '
           'numerical).')
    with pytest.raises(TypeError) as exin:
        ftsse._input_is_valid(
            np.array([[0, None], [0, 8]]), None, None, None, None, None)
    assert str(exin.value) == msg

    msg = ('Non-probabilistic functionality requires the global model to be '
           'capable of outputting probabilities via predict method.')
    model = InvalidModel()
    with pytest.raises(IncompatibleModelError) as exin:
        ftsse._input_is_valid(
            NUMERICAL_NP_ARRAY, model, False, None, None, None)
    assert str(exin.value) == msg

    msg = ('Probabilistic functionality requires the global model to be '
           'capable of outputting probabilities via predict_proba method.')
    model = NonProbaModel()
    with pytest.raises(IncompatibleModelError) as exin:
        ftsse._input_is_valid(
            NUMERICAL_NP_ARRAY, model, True, None, None, None)
    assert str(exin.value) == msg

    msg = ('The following indices are invalid for the input dataset: '
           '{}.'.format(np.array(['a'])))
    model = fum.KNN(k=3)
    with pytest.raises(IndexError) as exin:
        ftsse._input_is_valid(
            NUMERICAL_NP_ARRAY, model, True, ['a'], None, None)
    assert str(exin.value) == msg

    msg = ('The categorical_indices parameter must be a Python list or None.')
    with pytest.raises(TypeError) as exin:
        ftsse._input_is_valid(NUMERICAL_NP_ARRAY, model, True, 'a', None, None)
    assert str(exin.value) == msg

    msg = 'The class_names parameter must be None or a list.'
    with pytest.raises(TypeError) as exin:
        ftsse._input_is_valid(
            NUMERICAL_NP_ARRAY, model, True, None, 'a', None)
    assert str(exin.value) == msg

    msg = ('The class_name has to be either None or a string or a list of '
           'strings.')
    with pytest.raises(TypeError) as exin:
        ftsse._input_is_valid(
            NUMERICAL_NP_ARRAY, model, True, None, [0], None)
    assert str(exin.value) == msg

    msg = 'The feature_names parameter must be None or a list.'
    with pytest.raises(TypeError) as exin:
        ftsse._input_is_valid(
            NUMERICAL_NP_ARRAY, model, True, None, None, 'a')
    assert str(exin.value) == msg

    msg = ('The length of feature_names must be equal to the number of '
           'features in the dataset.')
    with pytest.raises(ValueError) as exin:
        ftsse._input_is_valid(
            NUMERICAL_NP_ARRAY, model, True, None, None, ['a'])
    assert str(exin.value) == msg

    msg = ('The feature name has to be either None or a string or a list of '
           'strings.')
    with pytest.raises(TypeError) as exin:
        ftsse._input_is_valid(
            NUMERICAL_NP_ARRAY, model, True, None, None, [0, 1, 2, 3])
    assert str(exin.value) == msg

    # All ok
    assert ftsse._input_is_valid(NUMERICAL_NP_ARRAY, model, True, [0],
                                 ['class 1'], ['1', '2', '3', '4'])


class TestSurrogateExplainer():
    """
    Tests :class:`fatf.transparency.sklearn.surrogate_explainers.\
    SurrogateExplainer` abstract class.
    """

    class BrokenSurrogateExplainer(ftsse.SurrogateExplainer):
        """
        A broken surrogate explainer implementation.

        This class does not have a ``explain_instance`` method.
        """

        def __init__(self, dataset, global_model, probabilistic,
                     categorical_indices, class_names, feature_names):
            """
            Dummy init method.
            """
            super().__init__(dataset, global_model, probabilistic,
                             categorical_indices, class_names, feature_names)

    class BaseSurrogateExplainer(ftsse.SurrogateExplainer):
        """
        A dummy surrogate explainer implementation.
        """

        def __init__(self, dataset, global_model, probabilistic=True,
                     categorical_indices=None, class_names=None,
                     feature_names=None):
            """
            Dummy init method.
            """
            super().__init__(
                dataset, global_model, probabilistic, categorical_indices,
                class_names, feature_names)

        def explain_instance(self, data_row):
            """
            Dummy explain_insatnce method.
            """
            self._explain_instance_input_is_valid(data_row)
            return {'0':{'0': 1}}

    numerical_np_array_classifier = fum.KNN(k=3)
    numerical_np_array_classifier.fit(NUMERICAL_NP_ARRAY,
                                      NUMERICAL_NP_ARRAY_TARGET)

    numerical_struct_array_classifier = fum.KNN(k=3)
    numerical_struct_array_classifier.fit(NUMERICAL_STRUCT_ARRAY,
                                          NUMERICAL_NP_ARRAY_TARGET)

    categorical_np_array_classifier = fum.KNN(k=3)
    categorical_np_array_classifier.fit(CATEGORICAL_NP_ARRAY,
                                        NUMERICAL_NP_ARRAY_TARGET)

    categorical_struct_array_classifier = fum.KNN(k=3)
    categorical_struct_array_classifier.fit(CATEGORICAL_STRUCT_ARRAY,
                                            NUMERICAL_NP_ARRAY_TARGET)

    mixed_classifier = fum.KNN(k=3)
    mixed_classifier.fit(MIXED_ARRAY, NUMERICAL_NP_ARRAY_TARGET)

    numerical_dummy_surrogate = BaseSurrogateExplainer(
        NUMERICAL_NP_ARRAY,
        numerical_np_array_classifier,
        probabilistic=True,
        categorical_indices=[0],
        class_names=None,
        feature_names=['1', '2', '3', '4'])

    numerical_struct_dummy_surrogate = BaseSurrogateExplainer(
        NUMERICAL_STRUCT_ARRAY,
        numerical_struct_array_classifier,
        probabilistic=False,
        categorical_indices=['a', 'b'],
        class_names=['class1', 'class2', 'class3'],
        feature_names=None)

    categorical_np_dummy_surrogate = BaseSurrogateExplainer(
        CATEGORICAL_NP_ARRAY,
        categorical_np_array_classifier,
        categorical_indices=[0, 1, 2])

    categorical_struct_dummy_surrogate = BaseSurrogateExplainer(
        CATEGORICAL_STRUCT_ARRAY,
        categorical_struct_array_classifier,
        categorical_indices=['a', 'b', 'c'],
        class_names=['class1', 'class2', 'class3'],
        feature_names=['1', '2', '3'])

    mixed_dummy_surrogate = BaseSurrogateExplainer(
        MIXED_ARRAY,
        mixed_classifier,
        categorical_indices=['b', 'd'],
        feature_names=['num1', 'str1', 'num2', 'str2'])

    def test_surrogate_explainer_init(self):
        """
        Tests :func:`fatf.transparency.sklearn.surrogate_edplainers.\
        SurrogateExplainer.__init__` initialiser.
        """
        abstract_method_error = ("Can't instantiate abstract class "
                                 "{} with abstract methods explain_instance")

        user_warning = (
            'Some of the string-based columns in the input dataset were not '
            'selected as categorical features via the categorical_indices '
            'parameter. String-based columns cannot be treated as numerical '
            'features, therefore they will be also treated as categorical '
            'features (in addition to the ones selected with the '
            'categorical_indices parameter).')

        # Don't define explain_instance method.
        with pytest.raises(TypeError) as exin:
            self.BrokenSurrogateExplainer(NUMERICAL_NP_ARRAY,
                                          self.numerical_np_array_classifier)
        msg = abstract_method_error.format('BrokenSurrogateExplainer')
        assert str(exin.value) == msg
        #
        with pytest.raises(TypeError) as exin:
            ftsse.SurrogateExplainer(NUMERICAL_NP_ARRAY,
                                     self.numerical_np_array_classifier)
        assert str(exin.value) == abstract_method_error.format(
            'SurrogateExplainer')

        # Warning for handling of categorical indices.
        with pytest.warns(UserWarning) as warning:
            surrogate_explainer = self.BaseSurrogateExplainer(
                CATEGORICAL_NP_ARRAY, self.categorical_np_array_classifier,
                True, [0])
        assert len(warning) == 1
        assert str(warning[0].message) == user_warning
        assert np.array_equal(surrogate_explainer.categorical_indices,
                              [0, 1, 2])
        #
        with pytest.warns(UserWarning) as warning:
            surrogate_explainer = self.BaseSurrogateExplainer(
                CATEGORICAL_STRUCT_ARRAY,
                self.categorical_struct_array_classifier,
                True,
                ['a'])
        assert len(warning) == 1
        assert str(warning[0].message) == user_warning
        assert np.array_equal(surrogate_explainer.categorical_indices,
                              np.array(['a', 'b', 'c']))
        #
        with pytest.warns(UserWarning) as warning:
            surrogate_explainer = self.BaseSurrogateExplainer(
                MIXED_ARRAY, self.mixed_classifier, True, ['b'])
        assert len(warning) == 1
        assert str(warning[0].message) == user_warning
        assert np.array_equal(surrogate_explainer.categorical_indices,
                              ['b', 'd'])

        # Validate internal variables
        assert np.array_equal(self.numerical_dummy_surrogate.dataset,
                              NUMERICAL_NP_ARRAY)
        assert not self.numerical_dummy_surrogate.is_structured
        assert self.numerical_dummy_surrogate.categorical_indices == [0]
        assert self.numerical_dummy_surrogate.numerical_indices == [1, 2, 3]
        assert self.numerical_dummy_surrogate.n_classes == 3
        assert list(self.numerical_dummy_surrogate.indices) == [0, 1, 2, 3]
        assert self.numerical_dummy_surrogate.class_names == \
            ['class 0', 'class 1', 'class 2']
        assert self.numerical_dummy_surrogate.feature_names == \
            ['1', '2', '3', '4']
        assert self.numerical_dummy_surrogate.prediction_function == \
            self.numerical_np_array_classifier.predict_proba

        assert np.array_equal(self.numerical_struct_dummy_surrogate.dataset,
                              NUMERICAL_STRUCT_ARRAY)
        assert self.numerical_struct_dummy_surrogate.is_structured
        assert self.numerical_struct_dummy_surrogate.categorical_indices == \
            ['a', 'b']
        assert self.numerical_struct_dummy_surrogate.numerical_indices == \
            ['c', 'd']
        assert self.numerical_struct_dummy_surrogate.n_classes == 3
        assert list(self.numerical_struct_dummy_surrogate.indices) == \
            ['a', 'b', 'c', 'd']
        assert self.numerical_struct_dummy_surrogate.class_names == \
            ['class1', 'class2', 'class3']
        assert self.numerical_struct_dummy_surrogate.feature_names == \
            ['feature 0', 'feature 1', 'feature 2', 'feature 3']
        assert self.numerical_struct_dummy_surrogate.prediction_function == \
            self.numerical_struct_array_classifier.predict

        assert np.array_equal(self.categorical_np_dummy_surrogate.dataset,
                              CATEGORICAL_NP_ARRAY)
        assert not self.categorical_np_dummy_surrogate.is_structured
        assert self.categorical_np_dummy_surrogate.categorical_indices == \
            [0, 1, 2]
        assert self.categorical_np_dummy_surrogate.numerical_indices == []
        assert self.categorical_np_dummy_surrogate.n_classes == 3
        assert list(self.categorical_np_dummy_surrogate.indices) == \
            [0, 1, 2]
        assert self.categorical_np_dummy_surrogate.class_names == \
            ['class 0', 'class 1', 'class 2']
        assert self.categorical_np_dummy_surrogate.feature_names == \
            ['feature 0', 'feature 1', 'feature 2']
        assert self.categorical_np_dummy_surrogate.prediction_function == \
            self.categorical_np_array_classifier.predict_proba

        assert np.array_equal(self.categorical_struct_dummy_surrogate.dataset,
                              CATEGORICAL_STRUCT_ARRAY)
        assert self.categorical_struct_dummy_surrogate.is_structured
        assert self.categorical_struct_dummy_surrogate.categorical_indices == \
            ['a', 'b', 'c']
        assert self.categorical_struct_dummy_surrogate.numerical_indices == []
        assert self.categorical_struct_dummy_surrogate.n_classes == 3
        assert list(self.categorical_struct_dummy_surrogate.indices) == \
            ['a', 'b', 'c']
        assert self.categorical_struct_dummy_surrogate.class_names == \
            ['class1', 'class2', 'class3']
        assert self.categorical_struct_dummy_surrogate.feature_names == \
            ['1', '2', '3']
        assert self.categorical_struct_dummy_surrogate.prediction_function == \
            self.categorical_struct_array_classifier.predict_proba

        assert np.array_equal(self.mixed_dummy_surrogate.dataset,
                              MIXED_ARRAY)
        assert self.mixed_dummy_surrogate.is_structured
        assert self.mixed_dummy_surrogate.categorical_indices == ['b', 'd']
        assert self.mixed_dummy_surrogate.numerical_indices == ['a', 'c']
        assert self.mixed_dummy_surrogate.n_classes == 3
        assert list(self.mixed_dummy_surrogate.indices) == \
            ['a', 'b', 'c', 'd']
        assert self.mixed_dummy_surrogate.class_names == \
            ['class 0', 'class 1', 'class 2']
        assert self.mixed_dummy_surrogate.feature_names == \
            ['num1', 'str1', 'num2', 'str2']
        assert self.mixed_dummy_surrogate.prediction_function == \
            self.mixed_classifier.predict_proba

    def test_explain_instance_validation(self):
        """
        Tests :func:`fatf.transparency.sklearn.surrogate_explainers.\
        SurrogateExplainer._explain_instance_is_valid` method.
        """
        incorrect_shape_data_row = ('The data_row must either be a '
                                    '1-dimensional numpy array or numpy void '
                                    'object for structured rows.')
        type_error_data_row = ('The dtype of the data_row is different to the '
                               'dtype of the data array used to initialise '
                               'this class.')
        incorrect_shape_features = ('The data_row must contain the same '
                                    'number of features as the dataset used '
                                    'to initialise this class.')

        # data_row shape
        with pytest.raises(IncorrectShapeError) as exin:
            self.numerical_dummy_surrogate.explain_instance(NUMERICAL_NP_ARRAY)
        assert str(exin.value) == incorrect_shape_data_row
        with pytest.raises(IncorrectShapeError) as exin:
            self.numerical_struct_dummy_surrogate.explain_instance(
                NUMERICAL_STRUCT_ARRAY)
        assert str(exin.value) == incorrect_shape_data_row

        # data_row type
        with pytest.raises(TypeError) as exin:
            self.numerical_dummy_surrogate.explain_instance(np.array(['a', 'b', 'c', 'd']))
        assert str(exin.value) == type_error_data_row
        with pytest.raises(TypeError) as exin:
            self.numerical_struct_dummy_surrogate.explain_instance(
                MIXED_ARRAY[0])
        assert str(exin.value) == type_error_data_row
        with pytest.raises(TypeError) as exin:
            self.categorical_np_dummy_surrogate.explain_instance(
                np.array([0.1]))
        assert str(exin.value) == type_error_data_row
        # Structured too short
        with pytest.raises(TypeError) as exin:
            self.numerical_struct_dummy_surrogate.explain_instance(
                MIXED_ARRAY[['a', 'b']][0])
        assert str(exin.value) == type_error_data_row

        # data_row features number
        with pytest.raises(IncorrectShapeError) as exin:
            self.numerical_dummy_surrogate.explain_instance(
                np.array([0.1, 1, 2]))
        assert str(exin.value) == incorrect_shape_features
        with pytest.raises(IncorrectShapeError) as exin:
            self.categorical_np_dummy_surrogate.explain_instance(
                np.array(['a', 'b']))
        assert str(exin.value) == incorrect_shape_features
