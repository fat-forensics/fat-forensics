"""
Holds custom distance functions used for FAT Forensics examples and testing.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import numpy as np
import pytest

import fatf.utils.models.models as fumm
from fatf.exceptions import (IncorrectShapeError, PrefittedModelError,
                             UnfittedModelError)


class TestModel(object):
    """
    Tests the :class:`fatf.utils.models.models.Model` abstract model class.
    """

    # pylint: disable=useless-object-inheritance

    def test__init__(self):
        """
        Tests the :func:``~fatf.utils.models.models.Model.__init__` method.
        """
        # pylint: disable=no-self-use
        error_message = ("Can't instantiate abstract class Model with "
                         'abstract methods __init__, fit, predict')
        with pytest.raises(TypeError) as exception_info:
            fumm.Model()  # pylint: disable=abstract-class-instantiated
        assert str(exception_info.value) == error_message

    def test_fit(self):
        """
        Tests the :func:`~fatf.utils.models.models.Model.fit` method.
        """
        # pylint: disable=no-self-use
        assert fumm.Model.fit(object(), np.ndarray((0, 0)),
                              np.ndarray((0, ))) is None  # yapf: disable

    def test_predict(self):
        """
        Tests the :func:`~fatf.utils.models.models.predict` method.
        """
        # pylint: disable=no-self-use
        assert fumm.Model.predict(object(), np.ndarray((0, ))) is None

    def test_predict_proba(self):
        """
        Tests the :func:`~fatf.utils.models.models.Model.predict_proba` method.
        """
        # pylint: disable=no-self-use
        error_message = ''
        with pytest.raises(NotImplementedError) as exception_info:
            fumm.Model.predict_proba(object(), np.ndarray((0, )))
        assert str(exception_info.value) == error_message


class TestKNN(object):
    """
    Tests :class:`fatf.utils.models.models.KNN` -- k-nearest neighbours model.
    """
    # pylint: disable=protected-access,useless-object-inheritance
    type_error_k = 'The k parameter has to be an integer.'
    value_error_k = 'The k parameter has to be a positive integer.'
    value_error_mode = ('The mode parameter has to have one of the following '
                        'values {}.').format(fumm.KNN._MODES)
    prefitted_model_error = 'This model has already been fitted.'
    incorrect_shape_error_rows = 'X must have at least one row.'
    incorrect_shape_error_2d = ('The training data must be a 2-dimensional '
                                'array.')
    incorrect_shape_error_1d = ('The training data labels must be a '
                                '1-dimensional array.')
    incorrect_shape_error_X0 = ('The data array has to have at least one data '
                                'point.')
    incorrect_shape_error_X1 = ('The data array has to have at least one '
                                'feature.')
    incorrect_shape_error_Xy = ('The number of samples in X must be the same '
                                'as the number of labels in y.')
    type_error_regressor = ('Regressor can only be fitted for a numerical '
                            'target vector.')
    unfitted_model_error = 'This model has not been fitted yet.'
    incorrect_shape_error_singular = ('X must be a 2-dimensional array. If '
                                      'you want to predict a single data '
                                      'point please format it as a single row '
                                      'in a 2-dimensional array.')
    value_error_dtype = 'X must have the same dtype as the training data.'
    incorrect_shape_error_columns = ('X must have the same number of columns '
                                     'as the training data (')
    runtime_error = 'This functionality is not available for a regressor.'

    k = 3

    X_short = np.array([[1, 2, 3], [1.4, 2.6, 3.8]])
    X_short_categorical_indices = np.array([])
    X_short_numerical_indices = np.array([0, 1, 2])
    X_short_n = 2

    y_short = np.array(['a', 'b'])
    # Lexicographical ordering
    short_majority_label = 'a'
    short_unique_y = np.array(['a', 'b'])
    short_unique_y_counts = np.array([1, 1])
    short_unique_y_probabilities = np.array([0.5, 0.5])

    y_short_numerical = np.array([3, 7])
    # Lexicographical ordering
    short_numerical_majority_label_classifier = 3
    short_numerical_majority_label_regressor = (3 + 7) / 2
    #
    short_numerical_unique_y = np.array([3, 7])
    short_numerical_unique_y_counts = np.array([1, 1])
    short_numerical_unique_y_probabilities = np.array([0.5, 0.5])

    X_cat = np.array([['a', 'b'], ['a', 'x'], ['yes', 'no'], ['5', '7'],
                      ['x', '0'], ['o', 'b']])
    X_cat_test = np.array([['yes', 'np'], ['yes', 'maybe'], ['x', '0']])
    X_cat_categorical_indices = np.array([0, 1])
    X_cat_numerical_indices = np.array([])
    X_cat_struct = np.array([('a', 'b'), ('a', 'x'), ('yes', 'no'), ('5', '7'),
                             ('x', '0'), ('o', 'b')],
                            dtype=[('x', '<U3'), ('y', '<U2')])
    X_cat_struct_test = np.array([('yes', 'np'), ('yes', 'maybe'), ('x', '0')],
                                 dtype=[('x', '<U3'), ('y', '<U5')])
    X_cat_struct_categorical_indices = np.array(['x', 'y'])
    X_cat_struct_numerical_indices = np.array([])
    X_cat_distances = np.array([[2, 2, 2],
                                [2, 2, 2],
                                [1, 1, 2],
                                [2, 2, 2],
                                [2, 2, 0],
                                [2, 2, 2]])  # yapf: disable

    X = np.array([[0, 0], [1, 1], [-1, 1], [-1, -1], [1, -1],
                  [2, 2], ])  # yapf: disable
    X_numerical_indices = np.array([0, 1])
    X_categorical_indices = np.array([])

    X_struct = np.array([(0, 0), (1, 1), (-1, 1), (-1, -1), (1, -1), (2, 2)],
                        dtype=[('a', int), ('b', 'f')])
    X_struct_numerical_indices = np.array(['a', 'b'])
    X_struct_categorical_indices = np.array([])

    X_mix = np.array([('a', 0, 0), ('b', 1, 1), ('c', -1, 1), ('c', -1, -1),
                      ('a', 1, -1), ('d', 2, 2)],
                     dtype=[('x', '<U6'), ('a', 'f'), ('b', 'f')])
    X_mix_numerical_indices = np.array(['a', 'b'])
    X_mix_categorical_indices = np.array(['x'])

    X_n = 6
    y = np.array([0, 1, 0, 0, 0, 1])
    y_categorical = np.array(['0', '1', '0', '0', '0', '1'])
    unique_y = np.array([0, 1])
    unique_y_categorical = np.array(['0', '1'])
    unique_y_counts = np.array([4, 2])
    unique_y_probabilities = np.array([0.667, 0.333])

    majority_label = 0
    majority_label_categorical = '0'
    majority_label_regressor = (1 + 1) / 6

    X_test = np.array([[-.5, -.5], [4, 4], [0, 2]])
    X_test_struct = np.array([(-.5, -.5), (4, 4), (0, 2)],
                             dtype=[('a', 'f'), ('b', float)])
    X_distances = np.array([[0.707, 5.657, 2.000],
                            [2.121, 4.243, 1.414],
                            [1.581, 5.831, 1.414],
                            [0.707, 7.071, 3.162],
                            [1.581, 5.831, 3.162],
                            [3.536, 2.828, 2.000]])  # yapf: disable
    y_test_3_classification = np.array([0, 1, 0])
    y_test_3_classification_categorical = np.array(['0', '1', '0'])
    y_test_3_regression = np.array([0, 0.667, 0.333])

    y_test_3_proba = np.array([[1, 0], [0.333, 0.667], [0.667, 0.333]])
    y_test_3_trainig_proba = np.array([0.667, 0.333])

    X_test_mix = np.array([('f', -.5, -.5), ('e', 4, 4), ('a', 0, 2)],
                          dtype=[('x', '<U6'), ('a', 'f'), ('b', 'f')])
    X_mix_distances = np.array([[1.707, 6.657, 2.000],
                                [3.121, 5.243, 2.414],
                                [2.581, 6.831, 2.414],
                                [1.707, 8.071, 4.162],
                                [2.581, 6.831, 3.162],
                                [4.536, 3.828, 3.000]])  # yapf: disable

    X_3D = np.ones((6, 2, 2))

    def _test_unfitted_internals(self,
                                 knn_clf,
                                 init_k=3,
                                 init_is_classifier=True):
        """
        Tests whether all internal attributes of an unfitted KNN model are OK.

        Parameters
        ----------
        knn_clf : object
            A KNN model to be tested.
        init_k : int
            The k parameter used to initialise the KNN model.
        init_is_classifier : boolean
            Whether the model was fitted as a classifier.
        """
        # pylint: disable=no-self-use
        assert knn_clf._k == init_k
        assert knn_clf._is_classifier is init_is_classifier

        assert knn_clf._is_fitted is False
        assert np.equal(knn_clf._X, np.ndarray((0, 0))).all()
        assert np.equal(knn_clf._y, np.ndarray((0, ))).all()
        assert knn_clf._X_n == int()
        assert np.equal(knn_clf._unique_y, np.ndarray((0, ))).all()
        assert np.equal(knn_clf._unique_y_counts, np.ndarray((0, ))).all()
        assert np.equal(knn_clf._unique_y_probabilities,
                        np.ndarray((0, ))).all()  # yapf: disable
        assert knn_clf._majority_label is None
        assert knn_clf._is_structured is False
        assert np.equal(knn_clf._categorical_indices, np.ndarray((0, ))).all()
        assert np.equal(knn_clf._numerical_indices, np.ndarray((0, ))).all()

    def _test_fitted_internals(self,
                               knn_clf,
                               is_structured,
                               X,
                               y,
                               X_n,
                               majority_label,
                               categorical_indices,
                               numerical_indices,
                               unique_y=None,
                               unique_y_counts=None,
                               unique_y_probabilities=None):
        """
        Tests whether all internal attributes of a fitted KNN model are OK.

        Parameters
        ----------
        knn_clf : object
            A KNN model to be tested.
        is_structured : boolean
            Indicates whether the model was trained on plane or structured
            numpy array.
        X : numpy.ndarray
            Used training data.
        y : numpy.ndarray
            Used training labels.
        X_n : integer
            Number of rows in ``X``.
        majority_label : Union[integer, string]
            The most frequent element in ``y``.
        categorical_indices : numpy.ndarray
            Categorical indices of the training array.
        numerical_indices : numpy.ndarray
            Numerical indices of the training array.
        unique_y : Optional[numpy.ndarray]
            Lexicographically ordered unique elements in ``y``.
        unique_y_counts : Optional[numpy.ndarray]
            Counts of the unique elements in ``y``.
        unique_y_probabilities : Optional[numpy.ndarray]
            Frequencies of the unique elements in ``y``.
        """
        # pylint: disable=no-self-use,too-many-arguments,invalid-name
        assert knn_clf._is_fitted
        assert np.array_equal(knn_clf._X, X)
        assert np.array_equal(knn_clf._y, y)
        assert knn_clf._X_n == X_n
        assert knn_clf._is_structured is is_structured
        assert np.array_equal(knn_clf._categorical_indices,
                              categorical_indices)
        assert np.array_equal(knn_clf._numerical_indices, numerical_indices)

        if knn_clf._is_classifier:
            assert np.array_equal(knn_clf._unique_y, unique_y)
            assert np.equal(knn_clf._unique_y_counts, unique_y_counts).all()
            assert np.isclose(
                knn_clf._unique_y_probabilities,
                unique_y_probabilities,
                atol=1e-3).all()
            assert knn_clf._majority_label == majority_label
        else:
            assert unique_y is None
            assert unique_y_counts is None
            assert unique_y_probabilities is None
            assert np.equal(knn_clf._unique_y, np.ndarray((0, ))).all()
            assert np.equal(knn_clf._unique_y_counts, np.ndarray((0, ))).all()
            assert np.equal(knn_clf._unique_y_probabilities,
                            np.ndarray((0, ))).all()  # yapf: disable
            assert knn_clf._majority_label == majority_label

    def test_knn(self):
        """
        Tests KNN initialisation.
        """
        # k is not an integer
        with pytest.raises(TypeError) as exception_info:
            clf = fumm.KNN(k=None)
        assert str(exception_info.value) == self.type_error_k
        with pytest.raises(TypeError) as exception_info:
            clf = fumm.KNN(k='k')
        assert str(exception_info.value) == self.type_error_k
        with pytest.raises(TypeError) as exception_info:
            clf = fumm.KNN(k=-5.5)
        assert str(exception_info.value) == self.type_error_k

        # k is a negative integer
        with pytest.raises(ValueError) as exception_info:
            clf = fumm.KNN(k=-5)
        assert str(exception_info.value) == self.value_error_k

        # mode specifier is wrong
        with pytest.raises(ValueError) as exception_info:
            clf = fumm.KNN(k=5, mode=object())
        assert str(exception_info.value) == self.value_error_mode
        with pytest.raises(ValueError) as exception_info:
            clf = fumm.KNN(k=5, mode=3)
        assert str(exception_info.value) == self.value_error_mode
        with pytest.raises(ValueError) as exception_info:
            clf = fumm.KNN(k=5, mode='C')
        assert str(exception_info.value) == self.value_error_mode

        clf = fumm.KNN()
        self._test_unfitted_internals(
            clf, init_k=self.k, init_is_classifier=True)

        k = 8
        clf = fumm.KNN(k=k)
        self._test_unfitted_internals(clf, init_k=k, init_is_classifier=True)

        clf = fumm.KNN(k=k, mode=None)
        self._test_unfitted_internals(clf, init_k=k, init_is_classifier=True)
        clf = fumm.KNN(k=k, mode='c')
        self._test_unfitted_internals(clf, init_k=k, init_is_classifier=True)
        clf = fumm.KNN(k=k, mode='classifier')
        self._test_unfitted_internals(clf, init_k=k, init_is_classifier=True)

        clf = fumm.KNN(k=k, mode='r')
        self._test_unfitted_internals(clf, init_k=k, init_is_classifier=False)
        clf = fumm.KNN(k=k, mode='regressor')
        self._test_unfitted_internals(clf, init_k=k, init_is_classifier=False)

    def test_fit(self):
        """
        Tests KNN fitting (:func:`~fatf.utils.models.models.KNN.fit`).
        """
        # pylint: disable=too-many-statements
        k = 2
        clf = fumm.KNN(k=k)
        self._test_unfitted_internals(clf, init_k=k)
        clf.fit(self.X, self.y)
        self._test_fitted_internals(
            clf, False, self.X, self.y, self.X_n, self.majority_label,
            self.X_categorical_indices, self.X_numerical_indices,
            self.unique_y, self.unique_y_counts, self.unique_y_probabilities)

        # Fitting a pre-fitted model
        with pytest.raises(PrefittedModelError) as exception_info:
            clf.fit(self.X, self.y)
        assert self.prefitted_model_error == str(exception_info.value)

        clf = fumm.KNN(k=k)
        self._test_unfitted_internals(clf, init_k=k)
        # X is not 2D
        with pytest.raises(IncorrectShapeError) as exception_info:
            clf.fit(self.X_3D, self.y)
        assert self.incorrect_shape_error_2d == str(exception_info.value)
        with pytest.raises(IncorrectShapeError) as exception_info:
            clf.fit(self.X_3D, self.X)
        assert self.incorrect_shape_error_2d == str(exception_info.value)
        # y is not 1D
        with pytest.raises(IncorrectShapeError) as exception_info:
            clf.fit(self.X, self.X_3D)
        assert self.incorrect_shape_error_1d == str(exception_info.value)

        # 0 examples
        with pytest.raises(IncorrectShapeError) as exception_info:
            clf.fit(np.ndarray((0, 5)), self.y)
        assert self.incorrect_shape_error_X0 == str(exception_info.value)
        with pytest.raises(IncorrectShapeError) as exception_info:
            clf.fit(np.ndarray((0, ), dtype=[('a', str), ('b', int)]), self.y)
        assert self.incorrect_shape_error_X0 == str(exception_info.value)

        # 0 features
        with pytest.raises(IncorrectShapeError) as exception_info:
            clf.fit(np.ndarray((5, 0)), self.y)
        assert self.incorrect_shape_error_X1 == str(exception_info.value)

        # Test whether the shape of X agrees with the shape of y
        with pytest.raises(IncorrectShapeError) as exception_info:
            clf.fit(self.X, self.X_numerical_indices)
        assert self.incorrect_shape_error_Xy == str(exception_info.value)

        # Fitting regressor to a categorical label vector
        clf = fumm.KNN(k=k, mode='r')
        self._test_unfitted_internals(clf, init_k=k, init_is_classifier=False)
        with pytest.raises(TypeError) as exception_info:
            y_pred = np.array(self.X.shape[0] * ['a'])
            clf.fit(self.X, y_pred)
        assert self.type_error_regressor == str(exception_info.value)

        # Fitting to a structured numerical array
        clf = fumm.KNN(k=k)
        self._test_unfitted_internals(clf, init_k=k, init_is_classifier=True)
        clf.fit(self.X_struct, self.y)
        self._test_fitted_internals(
            clf, True, self.X_struct, self.y, self.X_n, self.majority_label,
            self.X_struct_categorical_indices, self.X_struct_numerical_indices,
            self.unique_y, self.unique_y_counts, self.unique_y_probabilities)

        # Fitting to a structured mixed numerical-categorical array
        clf = fumm.KNN(k=k)
        self._test_unfitted_internals(clf, init_k=k, init_is_classifier=True)
        clf.fit(self.X_mix, self.y)
        self._test_fitted_internals(
            clf, True, self.X_mix, self.y, self.X_n, self.majority_label,
            self.X_mix_categorical_indices, self.X_mix_numerical_indices,
            self.unique_y, self.unique_y_counts, self.unique_y_probabilities)

        # Fit a regressor to a numerical data and check internal parameters
        clf = fumm.KNN(k=k, mode='regressor')
        self._test_unfitted_internals(clf, init_k=k, init_is_classifier=False)
        clf.fit(self.X_short, self.y_short_numerical)
        self._test_fitted_internals(
            clf, False, self.X_short, self.y_short_numerical, self.X_short_n,
            self.short_numerical_majority_label_regressor,
            self.X_short_categorical_indices, self.X_short_numerical_indices)

        # Fit a classifier to a numerical data and check internal parameters
        clf = fumm.KNN(k=k, mode='classifier')
        self._test_unfitted_internals(clf, init_k=k, init_is_classifier=True)
        clf.fit(self.X_short, self.y_short_numerical)
        self._test_fitted_internals(
            clf, False, self.X_short, self.y_short_numerical, self.X_short_n,
            self.short_numerical_majority_label_classifier,
            self.X_short_categorical_indices, self.X_short_numerical_indices,
            self.short_numerical_unique_y,
            self.short_numerical_unique_y_counts,
            self.short_numerical_unique_y_probabilities)

        # Fit a classifier to a categorical data and check internal parameters
        clf = fumm.KNN(k=k, mode='classifier')
        self._test_unfitted_internals(clf, init_k=k, init_is_classifier=True)
        clf.fit(self.X_short, self.y_short)
        self._test_fitted_internals(
            clf, False, self.X_short, self.y_short, self.X_short_n,
            self.short_majority_label, self.X_short_categorical_indices,
            self.X_short_numerical_indices, self.short_unique_y,
            self.short_unique_y_counts, self.short_unique_y_probabilities)

    def test_clear(self):
        """
        Tests KNN clearing (:func:`~fatf.utils.models.models.KNN.clear`).
        """
        k = 2
        clf = fumm.KNN(k=k)
        self._test_unfitted_internals(clf, init_k=k, init_is_classifier=True)

        # Clearing an unfitted model
        with pytest.raises(UnfittedModelError) as exception_info:
            clf.clear()
        assert self.unfitted_model_error == str(exception_info.value)

        # Clearing a fitted model
        clf.fit(self.X, self.y)
        self._test_fitted_internals(
            clf, False, self.X, self.y, self.X_n, self.majority_label,
            self.X_categorical_indices, self.X_numerical_indices,
            self.unique_y, self.unique_y_counts, self.unique_y_probabilities)
        clf.clear()
        self._test_unfitted_internals(clf, init_k=k, init_is_classifier=True)

    def test_get_distances(self):
        """
        Tests distances (:func:`~fatf.utils.models.models.KNN._get_distances`).
        """
        k = 2
        clf = fumm.KNN(k=k)

        def is_unfitted():
            return self._test_unfitted_internals(
                clf, init_k=2, init_is_classifier=True)

        is_unfitted()

        # Numerical distances on unstructured
        clf.fit(self.X, self.y)
        self._test_fitted_internals(
            clf, False, self.X, self.y, self.X_n, self.majority_label,
            self.X_categorical_indices, self.X_numerical_indices,
            self.unique_y, self.unique_y_counts, self.unique_y_probabilities)
        dist = clf._get_distances(self.X_test)
        assert np.isclose(dist, self.X_distances, atol=1e-3).all()

        # Categorical distances on unstructured
        clf.clear()
        is_unfitted()
        clf.fit(self.X_cat, self.y)
        self._test_fitted_internals(
            clf, False, self.X_cat, self.y, self.X_n, self.majority_label,
            self.X_cat_categorical_indices, self.X_cat_numerical_indices,
            self.unique_y, self.unique_y_counts, self.unique_y_probabilities)
        dist = clf._get_distances(self.X_cat_test)
        assert np.isclose(dist, self.X_cat_distances, atol=1e-3).all()

        # Numerical distances on structured
        clf.clear()
        is_unfitted()
        clf.fit(self.X_struct, self.y)
        self._test_fitted_internals(
            clf, True, self.X_struct, self.y, self.X_n, self.majority_label,
            self.X_struct_categorical_indices, self.X_struct_numerical_indices,
            self.unique_y, self.unique_y_counts, self.unique_y_probabilities)
        dist = clf._get_distances(self.X_test_struct)
        assert np.isclose(dist, self.X_distances, atol=1e-3).all()

        # Categorical distances on structured
        clf.clear()
        is_unfitted()
        clf.fit(self.X_cat_struct, self.y)
        self._test_fitted_internals(
            clf, True, self.X_cat_struct, self.y, self.X_n,
            self.majority_label, self.X_cat_struct_categorical_indices,
            self.X_cat_struct_numerical_indices, self.unique_y,
            self.unique_y_counts, self.unique_y_probabilities)
        dist = clf._get_distances(self.X_cat_struct_test)
        assert np.isclose(dist, self.X_cat_distances, atol=1e-3).all()

        # Numerical-categorical distances on structured
        clf.clear()
        is_unfitted()
        clf.fit(self.X_mix, self.y)
        self._test_fitted_internals(
            clf, True, self.X_mix, self.y, self.X_n, self.majority_label,
            self.X_mix_categorical_indices, self.X_mix_numerical_indices,
            self.unique_y, self.unique_y_counts, self.unique_y_probabilities)
        dist = clf._get_distances(self.X_test_mix)
        assert np.isclose(dist, self.X_mix_distances, atol=1e-3).all()

    def test_predict(self):
        """
        Tests KNN predictions (:func:`~fatf.utils.models.models.KNN.predict`).
        """
        # pylint: disable=too-many-statements
        k = 2
        clf = fumm.KNN(k=k)
        self._test_unfitted_internals(clf, init_k=k, init_is_classifier=True)

        # Unfitted model
        with pytest.raises(UnfittedModelError) as exception_info:
            clf.predict(self.X_test)
        assert self.unfitted_model_error == str(exception_info.value)

        # X is not 2D
        clf.fit(self.X, self.y)
        with pytest.raises(IncorrectShapeError) as exception_info:
            clf.predict(self.X_3D)
        assert self.incorrect_shape_error_singular == str(exception_info.value)
        with pytest.raises(IncorrectShapeError) as exception_info:
            clf.predict(self.y)
        assert self.incorrect_shape_error_singular == str(exception_info.value)

        # dtype is not similar to the training data
        with pytest.raises(ValueError) as exception_info:
            clf.predict(self.X_cat)
        assert self.value_error_dtype == str(exception_info.value)
        with pytest.raises(ValueError) as exception_info:
            clf.predict(self.X_cat_struct)
        assert self.value_error_dtype == str(exception_info.value)
        with pytest.raises(ValueError) as exception_info:
            clf.predict(self.X_struct)
        assert self.value_error_dtype == str(exception_info.value)

        # Predict 0 examples
        with pytest.raises(IncorrectShapeError) as exception_info:
            clf.predict(np.ones((0, 2), dtype=int))
        assert self.incorrect_shape_error_rows == str(exception_info.value)

        # The number of features disagrees...
        # ...unstructured
        with pytest.raises(IncorrectShapeError) as exception_info:
            clf.predict(self.X_distances)
        assert str(exception_info.value).startswith(
            self.incorrect_shape_error_columns)
        # ...structured
        clf.clear()
        self._test_unfitted_internals(clf, init_k=k, init_is_classifier=True)
        clf.fit(self.X_struct, self.y)
        self._test_fitted_internals(
            clf, True, self.X_struct, self.y, self.X_n, self.majority_label,
            self.X_struct_categorical_indices, self.X_struct_numerical_indices,
            self.unique_y, self.unique_y_counts, self.unique_y_probabilities)
        with pytest.raises(ValueError) as exception_info:
            clf.predict(self.X_test_struct[['a']])
        assert self.value_error_dtype == str(exception_info.value)

        # Regressor on unstructured
        # Sample smaller than k
        k = 3
        clf = fumm.KNN(k=k, mode='r')
        self._test_unfitted_internals(clf, init_k=k, init_is_classifier=False)
        clf.fit(self.X, self.y)
        self._test_fitted_internals(clf, False, self.X, self.y, self.X_n,
                                    self.majority_label_regressor,
                                    self.X_categorical_indices,
                                    self.X_numerical_indices)
        y_hat = clf.predict(self.X_test)
        assert np.isclose(y_hat, self.y_test_3_regression, atol=1e-3).all()
        # Sample bigger than k
        k = 10
        clf = fumm.KNN(k=k, mode='r')
        self._test_unfitted_internals(clf, init_k=k, init_is_classifier=False)
        clf.fit(self.X, self.y)
        self._test_fitted_internals(clf, False, self.X, self.y, self.X_n,
                                    self.majority_label_regressor,
                                    self.X_categorical_indices,
                                    self.X_numerical_indices)
        y_hat = clf.predict(self.X_test)
        y_true = np.array(y_hat.shape[0] * [self.majority_label_regressor])
        assert np.isclose(y_hat, y_true, atol=1e-3).all()

        # Regressor on structured
        # Sample smaller than k
        k = 3
        clf = fumm.KNN(k=k, mode='r')
        self._test_unfitted_internals(clf, init_k=k, init_is_classifier=False)
        clf.fit(self.X_struct, self.y)
        self._test_fitted_internals(clf, True, self.X_struct, self.y, self.X_n,
                                    self.majority_label_regressor,
                                    self.X_struct_categorical_indices,
                                    self.X_struct_numerical_indices)
        y_hat = clf.predict(self.X_test_struct)
        assert np.isclose(y_hat, self.y_test_3_regression, atol=1e-3).all()
        # Sample bigger than k
        k = 10
        clf = fumm.KNN(k=k, mode='r')
        self._test_unfitted_internals(clf, init_k=k, init_is_classifier=False)
        clf.fit(self.X_struct, self.y)
        self._test_fitted_internals(clf, True, self.X_struct, self.y, self.X_n,
                                    self.majority_label_regressor,
                                    self.X_struct_categorical_indices,
                                    self.X_struct_numerical_indices)
        y_hat = clf.predict(self.X_test_struct)
        y_true = np.array(y_hat.shape[0] * [self.majority_label_regressor])
        assert np.isclose(y_hat, y_true, atol=1e-3).all()

        # Numerical classifier on unstructured
        # Sample smaller than k
        k = 3
        clf = fumm.KNN(k=k)
        self._test_unfitted_internals(clf, init_k=k, init_is_classifier=True)
        clf.fit(self.X, self.y)
        self._test_fitted_internals(
            clf, False, self.X, self.y, self.X_n, self.majority_label,
            self.X_categorical_indices, self.X_numerical_indices,
            self.unique_y, self.unique_y_counts, self.unique_y_probabilities)
        y_hat = clf.predict(self.X_test)
        assert np.isclose(y_hat, self.y_test_3_classification, atol=1e-3).all()
        # Sample bigger than k
        k = 10
        clf = fumm.KNN(k=k)
        self._test_unfitted_internals(clf, init_k=k, init_is_classifier=True)
        clf.fit(self.X, self.y)
        self._test_fitted_internals(
            clf, False, self.X, self.y, self.X_n, self.majority_label,
            self.X_categorical_indices, self.X_numerical_indices,
            self.unique_y, self.unique_y_counts, self.unique_y_probabilities)
        y_hat = clf.predict(self.X_test)
        y_true = np.array(y_hat.shape[0] * [self.majority_label])
        assert np.isclose(y_hat, y_true, atol=1e-3).all()

        # Numerical classifier on structured
        # Sample smaller than k
        k = 3
        clf = fumm.KNN(k=k)
        self._test_unfitted_internals(clf, init_k=k, init_is_classifier=True)
        clf.fit(self.X_struct, self.y)
        self._test_fitted_internals(
            clf, True, self.X_struct, self.y, self.X_n, self.majority_label,
            self.X_struct_categorical_indices, self.X_struct_numerical_indices,
            self.unique_y, self.unique_y_counts, self.unique_y_probabilities)
        y_hat = clf.predict(self.X_test_struct)
        assert np.isclose(y_hat, self.y_test_3_classification, atol=1e-3).all()
        # Sample bigger than k
        k = 10
        clf = fumm.KNN(k=k)
        self._test_unfitted_internals(clf, init_k=k, init_is_classifier=True)
        clf.fit(self.X_struct, self.y)
        self._test_fitted_internals(
            clf, True, self.X_struct, self.y, self.X_n, self.majority_label,
            self.X_struct_categorical_indices, self.X_struct_numerical_indices,
            self.unique_y, self.unique_y_counts, self.unique_y_probabilities)
        y_hat = clf.predict(self.X_test_struct)
        y_true = np.array(y_hat.shape[0] * [self.majority_label])
        assert np.isclose(y_hat, y_true, atol=1e-3).all()

        # Categorical classifier on unstructured
        # Sample smaller than k
        k = 3
        clf = fumm.KNN(k=k)
        self._test_unfitted_internals(clf, init_k=k, init_is_classifier=True)
        clf.fit(self.X, self.y_categorical)
        self._test_fitted_internals(
            clf, False, self.X, self.y_categorical, self.X_n,
            self.majority_label_categorical, self.X_categorical_indices,
            self.X_numerical_indices, self.unique_y_categorical,
            self.unique_y_counts, self.unique_y_probabilities)
        y_hat = clf.predict(self.X_test)
        assert np.array_equal(y_hat, self.y_test_3_classification_categorical)
        # Sample bigger than k
        k = 10
        clf = fumm.KNN(k=k)
        self._test_unfitted_internals(clf, init_k=k, init_is_classifier=True)
        clf.fit(self.X, self.y_categorical)
        self._test_fitted_internals(
            clf, False, self.X, self.y_categorical, self.X_n,
            self.majority_label_categorical, self.X_categorical_indices,
            self.X_numerical_indices, self.unique_y_categorical,
            self.unique_y_counts, self.unique_y_probabilities)
        y_hat = clf.predict(self.X_test)
        y_true = np.array(y_hat.shape[0] * [self.majority_label_categorical])
        assert np.array_equal(y_hat, y_true)

        # Categorical classifier on structured
        # Sample smaller than k
        k = 3
        clf = fumm.KNN(k=k)
        self._test_unfitted_internals(clf, init_k=k, init_is_classifier=True)
        clf.fit(self.X_struct, self.y_categorical)
        self._test_fitted_internals(
            clf, True, self.X_struct, self.y_categorical, self.X_n,
            self.majority_label_categorical, self.X_struct_categorical_indices,
            self.X_struct_numerical_indices, self.unique_y_categorical,
            self.unique_y_counts, self.unique_y_probabilities)
        y_hat = clf.predict(self.X_test_struct)
        assert np.array_equal(y_hat, self.y_test_3_classification_categorical)
        # Sample bigger than k
        k = 10
        clf = fumm.KNN(k=k)
        self._test_unfitted_internals(clf, init_k=k, init_is_classifier=True)
        clf.fit(self.X_struct, self.y_categorical)
        self._test_fitted_internals(
            clf, True, self.X_struct, self.y_categorical, self.X_n,
            self.majority_label_categorical, self.X_struct_categorical_indices,
            self.X_struct_numerical_indices, self.unique_y_categorical,
            self.unique_y_counts, self.unique_y_probabilities)
        y_hat = clf.predict(self.X_test_struct)
        y_true = np.array(y_hat.shape[0] * [self.majority_label_categorical])
        assert np.array_equal(y_hat, y_true)

        # Test when the majority class is ambiguous -- sample smaller than k
        y = np.array([0, 1, 0, 1, 0, 1])  # pylint: disable=invalid-name
        majority_label = 0
        unique_y = np.array([0, 1])
        unique_y_counts = np.array([3, 3])
        unique_y_probabilities = np.array([.5, .5])
        X_test = np.array([[0, 0], [2, 0]])  # pylint: disable=invalid-name
        y_test = np.array([0, 0])
        #
        k = 4
        clf = fumm.KNN(k=k)
        self._test_unfitted_internals(clf, init_k=k, init_is_classifier=True)
        clf.fit(self.X, y)
        self._test_fitted_internals(clf, False, self.X, y, self.X_n,
                                    majority_label, self.X_categorical_indices,
                                    self.X_numerical_indices, unique_y,
                                    unique_y_counts, unique_y_probabilities)
        y_hat = clf.predict(X_test)
        assert np.array_equal(y_hat, y_test)

    def test_predict_proba(self):
        """
        Tests probas (:func:`~fatf.utils.models.models.KNN.predict_proba`).
        """
        # pylint: disable=too-many-statements
        # Regressor error
        k = 3
        clf = fumm.KNN(k=k, mode='r')
        self._test_unfitted_internals(clf, init_k=k, init_is_classifier=False)
        clf.fit(self.X, self.y)
        self._test_fitted_internals(clf, False, self.X, self.y, self.X_n,
                                    self.majority_label_regressor,
                                    self.X_categorical_indices,
                                    self.X_numerical_indices)
        with pytest.raises(RuntimeError) as exception_info:
            clf.predict_proba(self.X_test)
        assert str(exception_info.value) == self.runtime_error

        # Test other errors...
        k = 3
        clf = fumm.KNN(k=k)
        self._test_unfitted_internals(clf, init_k=k, init_is_classifier=True)

        # Unfitted model
        with pytest.raises(UnfittedModelError) as exception_info:
            clf.predict_proba(self.X_test)
        assert self.unfitted_model_error == str(exception_info.value)

        # ...
        clf.fit(self.X, self.y)
        self._test_fitted_internals(
            clf, False, self.X, self.y, self.X_n, self.majority_label,
            self.X_categorical_indices, self.X_numerical_indices,
            self.unique_y, self.unique_y_counts, self.unique_y_probabilities)

        # X is not 2D
        with pytest.raises(IncorrectShapeError) as exception_info:
            clf.predict_proba(self.X_3D)
        assert self.incorrect_shape_error_singular == str(exception_info.value)
        with pytest.raises(IncorrectShapeError) as exception_info:
            clf.predict_proba(self.y)
        assert self.incorrect_shape_error_singular == str(exception_info.value)

        # dtype is not similar to the training data
        with pytest.raises(ValueError) as exception_info:
            clf.predict_proba(self.X_cat)
        assert self.value_error_dtype == str(exception_info.value)
        with pytest.raises(ValueError) as exception_info:
            clf.predict_proba(self.X_cat_struct)
        assert self.value_error_dtype == str(exception_info.value)
        with pytest.raises(ValueError) as exception_info:
            clf.predict_proba(self.X_struct)
        assert self.value_error_dtype == str(exception_info.value)

        # Predict 0 examples
        with pytest.raises(IncorrectShapeError) as exception_info:
            clf.predict_proba(np.ones((0, 2), dtype=int))
        assert self.incorrect_shape_error_rows == str(exception_info.value)

        # The number of features disagrees...
        # ...unstructured
        with pytest.raises(IncorrectShapeError) as exception_info:
            clf.predict_proba(self.X_distances)
        assert str(exception_info.value).startswith(
            self.incorrect_shape_error_columns)
        # ...structured
        clf.clear()
        self._test_unfitted_internals(clf, init_k=k, init_is_classifier=True)
        clf.fit(self.X_struct, self.y)
        self._test_fitted_internals(
            clf, True, self.X_struct, self.y, self.X_n, self.majority_label,
            self.X_struct_categorical_indices, self.X_struct_numerical_indices,
            self.unique_y, self.unique_y_counts, self.unique_y_probabilities)
        with pytest.raises(ValueError) as exception_info:
            clf.predict_proba(self.X_test_struct[['a']])
        assert self.value_error_dtype == str(exception_info.value)

        # Numerical classifier on unstructured
        # Sample smaller than k
        k = 3
        clf = fumm.KNN(k=k)
        self._test_unfitted_internals(clf, init_k=k, init_is_classifier=True)
        clf.fit(self.X, self.y)
        self._test_fitted_internals(
            clf, False, self.X, self.y, self.X_n, self.majority_label,
            self.X_categorical_indices, self.X_numerical_indices,
            self.unique_y, self.unique_y_counts, self.unique_y_probabilities)
        y_hat = clf.predict_proba(self.X_test)
        assert np.isclose(y_hat, self.y_test_3_proba, atol=1e-3).all()
        # Sample bigger than k
        k = 10
        clf = fumm.KNN(k=k)
        self._test_unfitted_internals(clf, init_k=k, init_is_classifier=True)
        clf.fit(self.X, self.y)
        self._test_fitted_internals(
            clf, False, self.X, self.y, self.X_n, self.majority_label,
            self.X_categorical_indices, self.X_numerical_indices,
            self.unique_y, self.unique_y_counts, self.unique_y_probabilities)
        y_hat = clf.predict_proba(self.X_test)
        y_true = np.full(
            (y_hat.shape[0], self.y_test_3_trainig_proba.shape[0]),
            fill_value=self.y_test_3_trainig_proba)
        assert np.isclose(y_hat, y_true, atol=1e-3).all()

        # Numerical classifier on structured
        # Sample smaller than k
        k = 3
        clf = fumm.KNN(k=k)
        self._test_unfitted_internals(clf, init_k=k, init_is_classifier=True)
        clf.fit(self.X_struct, self.y)
        self._test_fitted_internals(
            clf, True, self.X_struct, self.y, self.X_n, self.majority_label,
            self.X_struct_categorical_indices, self.X_struct_numerical_indices,
            self.unique_y, self.unique_y_counts, self.unique_y_probabilities)
        y_hat = clf.predict_proba(self.X_test_struct)
        assert np.isclose(y_hat, self.y_test_3_proba, atol=1e-3).all()
        # Sample bigger than k
        k = 10
        clf = fumm.KNN(k=k)
        self._test_unfitted_internals(clf, init_k=k, init_is_classifier=True)
        clf.fit(self.X_struct, self.y)
        self._test_fitted_internals(
            clf, True, self.X_struct, self.y, self.X_n, self.majority_label,
            self.X_struct_categorical_indices, self.X_struct_numerical_indices,
            self.unique_y, self.unique_y_counts, self.unique_y_probabilities)
        y_hat = clf.predict_proba(self.X_test_struct)
        y_true = np.full(
            (y_hat.shape[0], self.y_test_3_trainig_proba.shape[0]),
            fill_value=self.y_test_3_trainig_proba)
        assert np.isclose(y_hat, y_true, atol=1e-3).all()

        # Categorical classifier on unstructured
        # Sample smaller than k
        k = 3
        clf = fumm.KNN(k=k)
        self._test_unfitted_internals(clf, init_k=k, init_is_classifier=True)
        clf.fit(self.X, self.y_categorical)
        self._test_fitted_internals(
            clf, False, self.X, self.y_categorical, self.X_n,
            self.majority_label_categorical, self.X_categorical_indices,
            self.X_numerical_indices, self.unique_y_categorical,
            self.unique_y_counts, self.unique_y_probabilities)
        y_hat = clf.predict_proba(self.X_test)
        assert np.isclose(y_hat, self.y_test_3_proba, atol=1e-3).all()
        # Sample bigger than k
        k = 10
        clf = fumm.KNN(k=k)
        self._test_unfitted_internals(clf, init_k=k, init_is_classifier=True)
        clf.fit(self.X, self.y_categorical)
        self._test_fitted_internals(
            clf, False, self.X, self.y_categorical, self.X_n,
            self.majority_label_categorical, self.X_categorical_indices,
            self.X_numerical_indices, self.unique_y_categorical,
            self.unique_y_counts, self.unique_y_probabilities)
        y_hat = clf.predict_proba(self.X_test)
        y_true = np.full(
            (y_hat.shape[0], self.y_test_3_trainig_proba.shape[0]),
            fill_value=self.y_test_3_trainig_proba)
        assert np.isclose(y_hat, y_true, atol=1e-3).all()

        # Categorical classifier on structured
        # Sample smaller than k
        k = 3
        clf = fumm.KNN(k=k)
        self._test_unfitted_internals(clf, init_k=k, init_is_classifier=True)
        clf.fit(self.X_struct, self.y_categorical)
        self._test_fitted_internals(
            clf, True, self.X_struct, self.y_categorical, self.X_n,
            self.majority_label_categorical, self.X_struct_categorical_indices,
            self.X_struct_numerical_indices, self.unique_y_categorical,
            self.unique_y_counts, self.unique_y_probabilities)
        y_hat = clf.predict_proba(self.X_test_struct)
        assert np.isclose(y_hat, self.y_test_3_proba, atol=1e-3).all()
        # Sample bigger than k
        k = 10
        clf = fumm.KNN(k=k)
        self._test_unfitted_internals(clf, init_k=k, init_is_classifier=True)
        clf.fit(self.X_struct, self.y_categorical)
        self._test_fitted_internals(
            clf, True, self.X_struct, self.y_categorical, self.X_n,
            self.majority_label_categorical, self.X_struct_categorical_indices,
            self.X_struct_numerical_indices, self.unique_y_categorical,
            self.unique_y_counts, self.unique_y_probabilities)
        y_hat = clf.predict_proba(self.X_test_struct)
        y_true = np.full(
            (y_hat.shape[0], self.y_test_3_trainig_proba.shape[0]),
            fill_value=self.y_test_3_trainig_proba)
        assert np.isclose(y_hat, y_true, atol=1e-3).all()
