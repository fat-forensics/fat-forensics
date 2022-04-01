"""
Tests the counterfactual prediction explainer of a black-box classifier.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import pytest

import numpy as np

import fatf.utils.models as fum
import fatf.transparency.predictions.counterfactuals as ftpc

from fatf.exceptions import IncorrectShapeError


def test_textualise_counterfactuals_errors():
    """
    Tests the ``textualise_counterfactuals`` function for errors.

    This function thests the :func:`fatf.transparency.predictions.
    counterfactuals.textualise_counterfactuals` function.
    """
    incorrect_shape_instance = ('The instance has to be a 1-dimensional numpy '
                                'array.')
    incorrect_shape_counterfactuals = ('The counterfactuals array should be a '
                                       '2-dimensional numpy array.')
    incorrect_shape_counterfactuals_distances = (
        'The counterfactuals_distances array should be a 1-dimensional array.')
    incorrect_shape_counterfactuals_predictions = (
        'The counterfactuals_predictions array should be a 1-dimensional '
        'array.')
    #
    value_error_instance = ('The instance has to be of a base type (strings '
                            'and/or numbers).')
    value_error_counterfactuals = ('The counterfactuals array has to be of a '
                                   'base type (strings and/or numbers).')
    value_error_counterfactuals_distances_type = (
        'The counterfactuals_distances array should be purely numerical.')
    value_error_counterfactuals_distances_shape = (
        'The counterfactuals_distances array should be of the same length as '
        'the number of rows in the counterfactuals array.')
    value_error_counterfactuals_predictions_type = (
        'The counterfactuals_predictions array should be of a base type '
        '(numbers and/or strings).')
    value_error_counterfactuals_predictions_shape = (
        'The counterfactuals_predictions array should be of the same length '
        'as the number of rows in the counterfactuals array.')
    value_error_counterfactuals_distances_instance = (
        'The type of the instance_class is different than the type of the '
        'counterfactuals_predictions array.')
    #
    type_error_instance_class = ('The instance_class has to be either an '
                                 'integer or a string.')
    type_error_intsance_counterfactuals = ('The type of the instance and the '
                                           'counterfactuals arrays do not '
                                           'agree.')
    #
    index_error_instance = ('The counterfactuals and instance column indices '
                            'do not agree. (The two arrays have different '
                            'number of columns.)')

    struct_array = np.array([('a', 1), ('b', 2)],
                            dtype=[('a', 'U1'), ('b', 'i')])
    classic_array = np.array([[0, 1], [5.5, 2]])
    classic_array_type = np.array([[0, None], [5.5, 2]])
    classic_array_str = np.array([[0, 'None'], [5.5, 2]])

    # Instance
    with pytest.raises(IncorrectShapeError) as exin:
        ftpc.textualise_counterfactuals(struct_array, struct_array)
    assert str(exin.value) == incorrect_shape_instance

    with pytest.raises(ValueError) as exin:
        ftpc.textualise_counterfactuals(classic_array_type[0], struct_array)
    assert str(exin.value) == value_error_instance

    # Counterfactuals
    with pytest.raises(IncorrectShapeError) as exin:
        ftpc.textualise_counterfactuals(struct_array[0], classic_array[0])
    assert str(exin.value) == incorrect_shape_counterfactuals

    with pytest.raises(ValueError) as exin:
        ftpc.textualise_counterfactuals(classic_array[0], classic_array_type)
    assert str(exin.value) == value_error_counterfactuals

    with pytest.raises(TypeError) as exin:
        ftpc.textualise_counterfactuals(struct_array[0], classic_array)
    assert str(exin.value) == type_error_intsance_counterfactuals

    # Instance class
    with pytest.raises(TypeError) as exin:
        ftpc.textualise_counterfactuals(classic_array[0], classic_array, 5.5)
    assert str(exin.value) == type_error_instance_class

    # Counterfactuals distances
    with pytest.raises(IncorrectShapeError) as exin:
        ftpc.textualise_counterfactuals(
            classic_array[0],
            classic_array,
            counterfactuals_distances=classic_array)
    assert str(exin.value) == incorrect_shape_counterfactuals_distances

    with pytest.raises(ValueError) as exin:
        ftpc.textualise_counterfactuals(
            classic_array[0],
            classic_array,
            counterfactuals_distances=classic_array_str[0])
    assert str(exin.value) == value_error_counterfactuals_distances_type

    with pytest.raises(ValueError) as exin:
        ftpc.textualise_counterfactuals(
            classic_array[0],
            classic_array,
            counterfactuals_distances=classic_array[1, [0]])
    assert str(exin.value) == value_error_counterfactuals_distances_shape

    # Counterfactuals predictions
    with pytest.raises(IncorrectShapeError) as exin:
        ftpc.textualise_counterfactuals(
            classic_array[0],
            classic_array,
            counterfactuals_predictions=classic_array)
    assert str(exin.value) == incorrect_shape_counterfactuals_predictions

    with pytest.raises(ValueError) as exin:
        ftpc.textualise_counterfactuals(
            classic_array[0],
            classic_array,
            counterfactuals_predictions=classic_array_type[0])
    assert str(exin.value) == value_error_counterfactuals_predictions_type

    with pytest.raises(ValueError) as exin:
        ftpc.textualise_counterfactuals(
            classic_array[0],
            classic_array,
            counterfactuals_predictions=classic_array[1, [0]])
    assert str(exin.value) == value_error_counterfactuals_predictions_shape

    with pytest.raises(ValueError) as exin:
        ftpc.textualise_counterfactuals(
            classic_array[0],
            classic_array,
            instance_class='1',
            counterfactuals_predictions=classic_array[1])
    assert str(exin.value) == value_error_counterfactuals_distances_instance

    # Invalid indices
    with pytest.raises(IndexError) as exin:
        ftpc.textualise_counterfactuals(classic_array[0],
                                        classic_array[:, [0]])
    assert str(exin.value) == index_error_instance


def test_textualise_counterfactuals():
    """
    Tests the ``textualise_counterfactuals`` function.

    This function thests the :func:`fatf.transparency.predictions.
    counterfactuals.textualise_counterfactuals` function.
    """
    instance_class_str = 'good'
    instance_class_int = 1
    counterfactuals_distances = np.array([4, 1])
    counterfactuals_predictions_str = np.array(['bad', 'mediocre'])
    counterfactuals_predictions_int = np.array([0, 2])
    classic_instance = np.array([1, 2, 3])
    classic_counterfactuals = np.array([[1, 2.5, 3.0], [0, 2, 5]])
    struct_instance = np.array([(1, 'bb', 3)],
                               dtype=[('a', 'i'), ('b', 'U2'), ('c', 'i')])[0]
    struct_counterfactuals = np.array(
        [(1, 'a', 3.0), (2, 'b', 3)],
        dtype=[('a', 'i'), ('b', 'U1'), ('c', 'i')]
    )  # yapf: disable
    cls_cf_0 = ('Instance:\n[1 2 3]\n\nFeature names: [0, 1, 2]\n\n'
                'Counterfactual instance:\n    feature *1*: *2* -> *2.5*\n\n'
                'Counterfactual instance:\n    feature *0*: *1* -> *0.0*\n'
                '    feature *2*: *3* -> *5.0*')
    cls_cf_1 = ('Instance (of class *good*):\n[1 2 3]\n\n'
                'Feature names: [0, 1, 2]\n\n'
                'Counterfactual instance (of class *mediocre*):\n'
                'Distance: 1\n'
                '    feature *0*: *1* -> *0.0*\n'
                '    feature *2*: *3* -> *5.0*\n\n'
                'Counterfactual instance (of class *bad*):\n'
                'Distance: 4\n'
                '    feature *1*: *2* -> *2.5*')
    stc_cf_0 = ('Instance (of class *1*):\n(1, \'bb\', 3)\n\n'
                'Feature names: (\'a\', \'b\', \'c\')\n\n'
                'Counterfactual instance (of class *0*):\n'
                '    feature *b*: *bb* -> *a*\n\n'
                'Counterfactual instance (of class *2*):\n'
                '    feature *a*: *1* -> *2*\n    feature *b*: *bb* -> *b*')

    # A minimal working example
    dsc = ftpc.textualise_counterfactuals(classic_instance,
                                          classic_counterfactuals)
    assert dsc == cls_cf_0

    # With the original class and counterfactual classes
    dsc = ftpc.textualise_counterfactuals(
        struct_instance,
        struct_counterfactuals,
        instance_class=instance_class_int,
        counterfactuals_predictions=counterfactuals_predictions_int)
    assert dsc == stc_cf_0

    # With the original class and counterfactual classes (strings) and dists
    dsc = ftpc.textualise_counterfactuals(
        classic_instance,
        classic_counterfactuals,
        instance_class=instance_class_str,
        counterfactuals_predictions=counterfactuals_predictions_str,
        counterfactuals_distances=counterfactuals_distances)
    assert dsc == cls_cf_1


class TestCounterfactualExplainer(object):
    """
    Tests the ``CounterfactualExplainer`` class.

    This function tests the :class:`fatf.transparency.predictions.
    counterfactuals.CounterfactualExplainer` class.
    """
    TARGET = np.array(
        ['good', 'good', 'bad', 'mediocre', 'bad', 'mediocre', 'good'])
    DATASET_NUM = np.array([[0, 0b1000, 35.70, 0b0011],
                            [1, 0b0100, 22.22, 0b0011],
                            [2, 0b0000, 11.11, 0b0011],
                            [3, 0b1101, 41.27, 0b0011],
                            [4, 0b0001, 12.57, 0b1100],
                            [5, 0b1111, 05.33, 0b1100],
                            [6, 0b0110, 17.29, 0b1100]])
    KNN_NUM = fum.KNN(k=1)
    KNN_NUM.fit(DATASET_NUM, TARGET)
    DATASET_STR = np.array([['a', 'a@a.com', '3', '0011'],
                            ['b', 'b@a.com', '2', '0011'],
                            ['c', 'c@a.com', '1', '0011'],
                            ['d', 'd@b.com', '4', '0011'],
                            ['e', 'e@b.com', '1', '1100'],
                            ['f', 'f@c.com', '0', '1100'],
                            ['g', 'g@d.com', '1', '1100']])
    KNN_STR = fum.KNN(k=1)
    KNN_STR.fit(DATASET_STR, TARGET)
    DATASET_STRUCT = np.array([('a', 'a@a.com', 35.70, '0011'),
                               ('b', 'b@a.com', 22.22, '0011'),
                               ('c', 'c@a.com', 11.11, '0011'),
                               ('d', 'd@b.com', 41.27, '0011'),
                               ('e', 'e@b.com', 12.57, '1100'),
                               ('f', 'f@c.com', 05.33, '1100'),
                               ('g', 'g@d.com', 17.29, '1100')],
                              dtype=[('name', 'U1'), ('email', 'U7'),
                                     ('q', float), ('postcode', 'U4')])
    KNN_STRUCT = fum.KNN(k=1)
    KNN_STRUCT.fit(DATASET_STRUCT, TARGET)

    @staticmethod
    def test_init_errors_one():
        """
        Tests some initialisation errors for the ``CounterfactualExplainer``.
        """
        user_warning_model = (
            'Model object characteristics are neither consistent with '
            'supervised nor unsupervised models.\n\n'
            '--> Unsupervised models <--\n'
            "The 'fit' method of the *Dummy* (model) class has incorrect "
            'number (2) of the required parameters. It needs to have exactly '
            '1 required parameter(s). Try using optional parameters if you '
            'require more functionality.\n'
            "The *Dummy* (model) class is missing 'predict' method.\n\n"
            '--> Supervised models <--\n'
            "The *Dummy* (model) class is missing 'predict' method.")
        user_warning_model_predictive_function = (
            'Both a model and a predictive_function parameters were supplied. '
            'A predictive functions takes the precedence during the '
            'execution.')
        #
        runtime_error_model = ('The model object requires a "predict" method '
                               'to be used with this explainer.')
        runtime_error_model_predictive_function = (
            'You either need to specify a model or a predictive_function '
            'parameter to initialise a counterfactual explainer.')
        type_error_predictive_function = ('The predictive_function parameter '
                                          'should be a Python function.')
        attribute_error_predictive_function = (
            'The predictive function requires exactly 1 non-optional '
            'parameter: a data array to be predicted.')
        value_error_dataset_type = ('The dataset has to be of a base type '
                                    '(strings and/or numbers).')
        incorrect_shape_dataset = 'The data array has to be 2-dimensional.'
        type_error_categorical_indices = ('categorical_indices parameter '
                                          'either has to be a list of indices '
                                          'or None.')
        type_error_numerical_indices = ('numerical_indices parameter either '
                                        'has to be a list of indices or None.')
        value_error_empty_indices = (
            'Both categorical_indices and numerical_indices parameters cannot '
            'be empty lists. If you want them to be inferred from a data '
            'array please leave these parameters set to None.')
        value_error_overlaping_indices = (
            'Some of the indices in the categorical_indices and '
            'numerical_indices parameters are repeated.')
        type_error_mixed_type_indices = (
            'Some of the indices given in the categorical_indices and/or '
            'numerical_indices parameters do not share the same type. It is '
            'expected that indices for a classic numpy array will all be '
            'integers and for a structured numpy array they will be strings.')

        # Not a functional model
        class Dummy(object):
            def __init__(self):
                pass

            def fit(self, X, y):
                pass  # pragma: nocover

        with pytest.warns(UserWarning) as w:
            with pytest.raises(RuntimeError) as exin:
                ftpc.CounterfactualExplainer(model=Dummy())
            assert str(exin.value) == runtime_error_model
        assert len(w) == 1
        assert str(w[0].message) == user_warning_model

        # Incorrect predictive function
        with pytest.raises(TypeError) as exin:
            ftpc.CounterfactualExplainer(predictive_function='func')
        assert str(exin.value) == type_error_predictive_function

        def func(i, j):
            return i + j  # pragma: nocover

        with pytest.raises(AttributeError) as exin:
            ftpc.CounterfactualExplainer(predictive_function=func)
        assert str(exin.value) == attribute_error_predictive_function

        # No model, no predictive function
        with pytest.raises(RuntimeError) as exin:
            ftpc.CounterfactualExplainer()
        assert str(exin.value) == runtime_error_model_predictive_function

        # Model and predictive function + non-base dataset
        knn = fum.KNN()
        with pytest.warns(UserWarning) as w:
            with pytest.raises(ValueError) as exin:
                ftpc.CounterfactualExplainer(
                    model=knn,
                    predictive_function=knn.predict,
                    dataset=np.array([0, None]))
            assert str(exin.value) == value_error_dataset_type
        assert len(w) == 1
        assert str(w[0].message) == user_warning_model_predictive_function

        # Non-2-D dataset
        with pytest.raises(IncorrectShapeError) as exin:
            ftpc.CounterfactualExplainer(model=knn, dataset=np.array([0, 1]))
        assert str(exin.value) == incorrect_shape_dataset

        # Categorical indices
        with pytest.raises(TypeError) as exin:
            ftpc.CounterfactualExplainer(model=knn, categorical_indices=2)
        assert str(exin.value) == type_error_categorical_indices

        # Numerical indices
        with pytest.raises(TypeError) as exin:
            ftpc.CounterfactualExplainer(model=knn, numerical_indices=2)
        assert str(exin.value) == type_error_numerical_indices

        # Categorical and numerical indices
        with pytest.raises(ValueError) as exin:
            ftpc.CounterfactualExplainer(
                model=knn, numerical_indices=[], categorical_indices=[])
        assert str(exin.value) == value_error_empty_indices
        #
        with pytest.raises(ValueError) as exin:
            ftpc.CounterfactualExplainer(
                model=knn,
                numerical_indices=[0, 1],
                categorical_indices=[1, 2])
        assert str(exin.value) == value_error_overlaping_indices
        #
        with pytest.raises(TypeError) as exin:
            ftpc.CounterfactualExplainer(
                model=knn,
                numerical_indices=[0, 1],
                categorical_indices=['1', '2'])
        assert str(exin.value) == type_error_mixed_type_indices

    @staticmethod
    def test_init_errors_middle():
        """
        Tests some initialisation errors for the ``CounterfactualExplainer``.
        """
        index_error_numerical = ('Some of the numerical_indices are not valid '
                                 'for the given array.')
        index_error_categorical = ('Some of the categorical_indices are not '
                                   'valid for the given array.')
        index_error_all_series = ('The union of categorical and numerical '
                                  'indices does not form a series of '
                                  'consecutive integers. This is required for '
                                  'an classic (unstructured) numpy array.')
        index_error_num_cat_incomplete = (
            'The numerical_indices and the categorical_indices parameters do '
            'not cover all of the columns in the given dataset array.')
        value_error_no_indices = ('If a dataset is not given, both '
                                  'categorical_indices and numerical_indices '
                                  'parameters have to be defined.')
        value_error_categorical_cat = (
            'Some of the categorical indices (textual columns) in the array '
            'were not indicated to be categorical by the user. Textual '
            'columns must not be treated as numerical features.')
        value_error_categorical_num = (
            'Some of the categorical fields in the input data set were '
            'indicated to be numerical indices via the numerical_indices '
            'parameter. Textual columns must not be treated as numerical '
            'features.')
        value_error_feature_range_fill = (
            'A dataset is needed to fill in feature ranges for features '
            'selected for counterfactuals that were not provided with ranges '
            'via feature_ranges parameter. If you do not want to provide a '
            'dataset please specify counterfactual feature ranges via '
            'feature_ranges parameter.')

        knn = fum.KNN()
        dataset = np.array([[0, 1, 2, 3, 4]])
        dataset_struct = np.array([(0, 1, 2, 3, '4')],
                                  dtype=[('a', 'i'), ('b', 'i'), ('c', 'i'),
                                         ('d', 'i'), ('e', 'U1')])

        with pytest.raises(IndexError) as exin:
            ftpc.CounterfactualExplainer(
                model=knn, dataset=dataset, numerical_indices=[5])
        assert str(exin.value) == index_error_numerical

        with pytest.raises(IndexError) as exin:
            ftpc.CounterfactualExplainer(
                model=knn, dataset=dataset_struct, categorical_indices=['f'])
        assert str(exin.value) == index_error_categorical

        with pytest.raises(IndexError) as exin:
            ftpc.CounterfactualExplainer(
                model=knn,
                categorical_indices=[1, 2, 5],
                numerical_indices=[4, 6])
        assert str(exin.value) == index_error_all_series

        with pytest.raises(ValueError) as exin:
            ftpc.CounterfactualExplainer(
                model=knn, categorical_indices=[1, 2, 5])
        assert str(exin.value) == value_error_no_indices

        with pytest.raises(ValueError) as exin:
            ftpc.CounterfactualExplainer(
                model=knn,
                dataset=dataset_struct,
                categorical_indices=['c', 'd'])
        assert str(exin.value) == value_error_categorical_cat

        with pytest.raises(ValueError) as exin:
            ftpc.CounterfactualExplainer(
                model=knn,
                dataset=dataset_struct,
                numerical_indices=['d', 'e'])
        assert str(exin.value) == value_error_categorical_num

        with pytest.raises(IndexError) as exin:
            ftpc.CounterfactualExplainer(
                model=knn,
                dataset=dataset_struct,
                numerical_indices=['a', 'c'],
                categorical_indices=['d', 'e', 'f'])
        assert str(exin.value) == index_error_num_cat_incomplete

        # Feature ranges fill-in
        with pytest.raises(ValueError) as exin:
            ftpc.CounterfactualExplainer(
                model=knn,
                numerical_indices=['a', 'b', 'c'],
                categorical_indices=['d', 'e'])
        assert str(exin.value) == value_error_feature_range_fill

    @staticmethod
    def test_init_errors_two():
        """
        Tests some initialisation errors for the ``CounterfactualExplainer``.
        """
        type_error_cf_indices = ('counterfactual_feature_indices parameter '
                                 'either has to be a list of indices or None.')
        value_error_cf_indices = (
            'counterfactual_feature_indices parameter cannot be an empty '
            'list. If you want all of the features to be used for '
            'counterfactuals generation leave this parameter unset or set it '
            'explicitly to None.')
        index_error_cf_indices = (
            'counterfactual_feature_indices list contains invalid indices.')
        type_error_max_cf_len = ('The max_counterfactual_length parameter '
                                 'should be an integer.')
        value_error_max_cf_len = (
            'The max_counterfactual_length parameter should be a non-negative '
            'integer. If you want to generate counterfactuals with a full '
            'length (number of features), set this parameter to 0.')
        type_error_feature_ranges = ('The feature_ranges parameter has to be '
                                     'a dictionary or None.')
        value_error_feature_ranges = ('The feature_ranges parameter cannot be '
                                      'an empty dictionary.')
        index_error_feature_ranges = ('Some of the indices (dictionary keys) '
                                      'in the feature_ranges parameter are '
                                      'not valid.')
        type_error_feature_ranges_cat = ('Categorical column range should be '
                                         'a list of values to be used for the '
                                         'counterfactuals generation process.')
        type_error_feature_ranges_cat_inst = ('The possible values defined '
                                              'for the *{}* feature do not '
                                              'share the same type.')
        value_error_feature_ranges_cat = ('A list specifying the possible '
                                          'values of a categorical feature '
                                          'should not be empty.')
        type_error_feature_ranges_num = ('Numerical column range should be a '
                                         'pair of numbers defining the lower '
                                         'and the upper limits of the range.')
        type_error_feature_ranges_num_inst = ('Both the lower and the upper '
                                              "bound defining column's range "
                                              'should numbers.')
        value_error_feature_ranges_num = ('Numerical column range tuple '
                                          'should just contain 2 numbers: the '
                                          'lower and the upper bounds of the '
                                          'range to be searched.')
        value_error_feature_ranges_num_inst = ('The second element of a tuple '
                                               'defining a numerical range '
                                               'should be strictly larger '
                                               'than the first element.')
        type_error_dist_func = ('The distance_functions parameter has to be a '
                                'dictionary.')
        value_error_dist_func = ('The distance_functions parameter cannot be '
                                 'an empty dictionary.')
        index_error_dist_func = ('Some of the indices (dictionary keys) in '
                                 'the distance_functions parameter are '
                                 'invalid.')
        type_error_dist_func_inst = ('All of the distance functions defined '
                                     'via the distance_functions parameter '
                                     'have to be Python callable.')
        attribute_error_dist_func = ('Every distance function requires '
                                     'exactly 2 non-optional parameters.')
        type_error_step_sizes = ('The step_sizes parameter has to be a '
                                 'dictionary.')
        value_error_step_sizes = ('The step_sizes parameter cannot be an '
                                  'empty dictionary.')
        index_error_step_sizes = ('Some of the indices (dictionary keys) in '
                                  'the step_sizes parameter are not valid.')
        type_error_step_sizes_inst = ('All of the step values contained in '
                                      'the step_sizes must be numbers.')
        value_error_step_sizes_inst = ('All of the step values contained in '
                                       'the step_sizes must be positive '
                                       'numbers.')
        type_error_default_num_ss = ('The default_numerical_step_size '
                                     'parameter has to be a number.')
        value_error_default_num_ss = ('The default_numerical_step_size '
                                      'parameter has to be a positive number.')

        knn = fum.KNN()
        dataset_struct = np.array([(0, 1, 2, 3, '4')],
                                  dtype=[('a', 'i'), ('b', 'i'), ('c', 'i'),
                                         ('d', 'i'), ('e', 'U1')])

        # Counterfactual feature indices
        with pytest.raises(TypeError) as exin:
            ftpc.CounterfactualExplainer(
                model=knn,
                dataset=dataset_struct,
                counterfactual_feature_indices='a')
        assert str(exin.value) == type_error_cf_indices

        with pytest.raises(ValueError) as exin:
            ftpc.CounterfactualExplainer(
                model=knn,
                dataset=dataset_struct,
                counterfactual_feature_indices=[])
        assert str(exin.value) == value_error_cf_indices

        with pytest.raises(IndexError) as exin:
            ftpc.CounterfactualExplainer(
                model=knn,
                dataset=dataset_struct,
                counterfactual_feature_indices=[1])
        assert str(exin.value) == index_error_cf_indices

        # Maximum counterfactual length
        with pytest.raises(TypeError) as exin:
            ftpc.CounterfactualExplainer(
                model=knn,
                dataset=dataset_struct,
                max_counterfactual_length=8.0)
        assert str(exin.value) == type_error_max_cf_len

        with pytest.raises(ValueError) as exin:
            ftpc.CounterfactualExplainer(
                model=knn,
                dataset=dataset_struct,
                max_counterfactual_length=-3)
        assert str(exin.value) == value_error_max_cf_len

        # Feature ranges
        with pytest.raises(TypeError) as exin:
            ftpc.CounterfactualExplainer(
                model=knn, dataset=dataset_struct, feature_ranges=[])
        assert str(exin.value) == type_error_feature_ranges

        with pytest.raises(ValueError) as exin:
            ftpc.CounterfactualExplainer(
                model=knn, dataset=dataset_struct, feature_ranges={})
        assert str(exin.value) == value_error_feature_ranges

        with pytest.raises(IndexError) as exin:
            ftpc.CounterfactualExplainer(
                model=knn, dataset=dataset_struct, feature_ranges={'f': 7})
        assert str(exin.value) == index_error_feature_ranges

        with pytest.raises(TypeError) as exin:
            ftpc.CounterfactualExplainer(
                model=knn, dataset=dataset_struct, feature_ranges={'e': 7})
        assert str(exin.value) == type_error_feature_ranges_cat

        with pytest.raises(ValueError) as exin:
            ftpc.CounterfactualExplainer(
                model=knn, dataset=dataset_struct, feature_ranges={'e': []})
        assert str(exin.value) == value_error_feature_ranges_cat

        with pytest.raises(TypeError) as exin:
            ftpc.CounterfactualExplainer(
                model=knn,
                dataset=dataset_struct,
                feature_ranges={'e': [7, 8, 'a']})
        assert str(
            exin.value) == type_error_feature_ranges_cat_inst.format('e')

        with pytest.raises(TypeError) as exin:
            ftpc.CounterfactualExplainer(
                model=knn,
                dataset=dataset_struct,
                feature_ranges={'b': (0, 5), 'c': 7})  # yapf: disable
        assert str(exin.value) == type_error_feature_ranges_num

        with pytest.raises(ValueError) as exin:
            ftpc.CounterfactualExplainer(
                model=knn,
                dataset=dataset_struct,
                feature_ranges={'b': (0, 5), 'c': (0, 5, 9)})  # yapf: disable
        assert str(exin.value) == value_error_feature_ranges_num

        with pytest.raises(TypeError) as exin:
            ftpc.CounterfactualExplainer(
                model=knn,
                dataset=dataset_struct,
                feature_ranges={'b': (0, 5), 'c': (0, 's')})  # yapf: disable
        assert str(exin.value) == type_error_feature_ranges_num_inst

        with pytest.raises(ValueError) as exin:
            ftpc.CounterfactualExplainer(
                model=knn,
                dataset=dataset_struct,
                feature_ranges={'b': (0, 5), 'c': (5, 5.0)})  # yapf: disable
        assert str(exin.value) == value_error_feature_ranges_num_inst

        # Distance functions
        with pytest.raises(TypeError) as exin:
            ftpc.CounterfactualExplainer(
                model=knn, dataset=dataset_struct, distance_functions='b')
        assert str(exin.value) == type_error_dist_func

        with pytest.raises(ValueError) as exin:
            ftpc.CounterfactualExplainer(
                model=knn, dataset=dataset_struct, distance_functions={})
        assert str(exin.value) == value_error_dist_func

        with pytest.raises(IndexError) as exin:
            ftpc.CounterfactualExplainer(
                model=knn, dataset=dataset_struct, distance_functions={'h': 7})
        assert str(exin.value) == index_error_dist_func

        with pytest.raises(TypeError) as exin:
            ftpc.CounterfactualExplainer(
                model=knn, dataset=dataset_struct, distance_functions={'e': 7})
        assert str(exin.value) == type_error_dist_func_inst

        def unattribed(x):
            return x  # pragma: nocover

        with pytest.raises(AttributeError) as exin:
            ftpc.CounterfactualExplainer(
                model=knn,
                dataset=dataset_struct,
                distance_functions={'e': unattribed})
        assert str(exin.value) == attribute_error_dist_func

        # Step sizes
        with pytest.raises(TypeError) as exin:
            ftpc.CounterfactualExplainer(
                model=knn, dataset=dataset_struct, step_sizes=7)
        assert str(exin.value) == type_error_step_sizes

        with pytest.raises(ValueError) as exin:
            ftpc.CounterfactualExplainer(
                model=knn, dataset=dataset_struct, step_sizes={})
        assert str(exin.value) == value_error_step_sizes

        with pytest.raises(IndexError) as exin:
            ftpc.CounterfactualExplainer(
                model=knn, dataset=dataset_struct, step_sizes={'zz': 'tops'})
        assert str(exin.value) == index_error_step_sizes

        with pytest.raises(TypeError) as exin:
            ftpc.CounterfactualExplainer(
                model=knn, dataset=dataset_struct, step_sizes={'a': 'tops'})
        assert str(exin.value) == type_error_step_sizes_inst

        with pytest.raises(ValueError) as exin:
            ftpc.CounterfactualExplainer(
                model=knn, dataset=dataset_struct, step_sizes={'a': 0})
        assert str(exin.value) == value_error_step_sizes_inst

        # Default numerical step size
        with pytest.raises(TypeError) as exin:
            ftpc.CounterfactualExplainer(
                model=knn,
                dataset=dataset_struct,
                default_numerical_step_size='a')
        assert str(exin.value) == type_error_default_num_ss

        with pytest.raises(ValueError) as exin:
            ftpc.CounterfactualExplainer(
                model=knn,
                dataset=dataset_struct,
                default_numerical_step_size=-7)
        assert str(exin.value) == value_error_default_num_ss

    @staticmethod
    def test_init_errors_auto_range():
        """
        Tests for errors in range calculations when initialising the class.
        """
        user_warning = ('There is only one unique value detected for the '
                        'categorical feature *{}*: {}.')
        value_error = ('The minimum and the maximum detected value for '
                       'feature *{}* are the same ({}). Impossible to create '
                       'a range.')
        # Categorical type generalisation
        data = np.array([[0, 0], [1, 0]])
        clf = fum.KNN(k=1)
        clf.fit(data, np.array((['good', 'bad'])))

        with pytest.raises(ValueError) as exin:
            ftpc.CounterfactualExplainer(model=clf, dataset=data)
        assert str(exin.value) == value_error.format(1, 0)

        with pytest.warns(UserWarning) as w:
            ftpc.CounterfactualExplainer(
                model=clf, dataset=data, categorical_indices=[1])
        assert len(w) == 1
        assert str(w[0].message) == user_warning.format(1, '[0]')

    def test_explain_instance_errors(self):
        """
        Tests for error in ``explain_instance`` method.
        """
        incorrect_shape_error = ('The instance to be explained should be a '
                                 '1-dimensional numpy array or a row of a '
                                 'structured array (numpy.void).')
        value_error_type = ('The instance should be of a base type -- a '
                            'mixture of numerical and textual types.')
        index_error = ('The indices used to initialise this class are not '
                       'valid for this data point.')
        type_error_cf_class = ('The counterfactual class should be either an '
                               'integer or a string.')
        type_error_normalise = ('The normalise_distance parameter should be a '
                                'boolean.')

        cfe = ftpc.CounterfactualExplainer(
            model=self.KNN_NUM, dataset=self.DATASET_NUM)

        with pytest.raises(IncorrectShapeError) as exin:
            cfe.explain_instance(self.DATASET_NUM)
        assert str(exin.value) == incorrect_shape_error

        with pytest.raises(ValueError) as exin:
            cfe.explain_instance(np.array([None, 42]))
        assert str(exin.value) == value_error_type

        with pytest.raises(IndexError) as exin:
            cfe.explain_instance(np.array([0]))
        assert str(exin.value) == index_error

        with pytest.raises(TypeError) as exin:
            cfe.explain_instance(
                np.array([0, 7, 4, 2]), counterfactual_class=7.0)
        assert str(exin.value) == type_error_cf_class

        with pytest.raises(TypeError) as exin:
            cfe.explain_instance(np.array([0, 7, 4, 2]), normalise_distance=1)
        assert str(exin.value) == type_error_normalise

    def test_counterfactuals_automatic(self):
        """
        Tests counterfactuals generation with the ``CounterfactualExplainer``.

        The parameters will be "auto-tuned".
        """
        cfe = ftpc.CounterfactualExplainer(
            model=self.KNN_NUM,
            dataset=self.DATASET_NUM,
            categorical_indices=[0, 1, 3])
        cfs, cfs_dist, cfs_pred = cfe.explain_instance(self.DATASET_NUM[2])
        #
        t_cfs = np.array([[2, 15, 11.11, 3], [2, 13, 11.11, 3],
                          [2, 15, 11.33, 3], [2, 13, 11.33, 3],
                          [5, 15, 11.11, 3], [5, 13, 11.11, 3],
                          [4, 15, 11.11, 3], [4, 13, 11.11, 3],
                          [3, 15, 11.11, 3], [3, 13, 11.11, 3],
                          [2, 15, 11.11, 12], [6, 15, 11.11, 3],
                          [2, 13, 11.11, 12], [1, 15, 11.11, 3],
                          [1, 13, 11.11, 3], [0, 15, 11.11, 3],
                          [0, 13, 11.11, 3], [6, 13, 11.11, 3],
                          [2, 0, 18.33, 3], [0, 0, 17.33, 3],
                          [2, 0, 20.33, 12]])  # yapf: disable
        t_dist = np.array([1, 1, 1.22, 1.22] + 14 * [2] + [7.22, 7.22, 10.22])
        t_pred = np.array(18 * ['mediocre'] + 3 * ['good'])
        assert np.array_equal(cfs_pred, t_pred)
        assert np.allclose(cfs_dist, t_dist)
        uid = [[0, 1], [2, 3], list(range(4, 18)), [18, 19], [20]]
        for i in uid:
            srtd1 = np.sort(cfs[i], axis=0)
            srtd2 = np.sort(t_cfs[i], axis=0)
            assert np.array_equal(srtd1, srtd2)

        cfe = ftpc.CounterfactualExplainer(
            predictive_function=self.KNN_STR.predict, dataset=self.DATASET_STR)
        cfs, cfs_dist, cfs_pred = cfe.explain_instance(
            self.DATASET_STR[2], counterfactual_class='mediocre')
        #
        t_cfs = np.array([['c', 'd@b.com', '4', '0011'],
                          ['d', 'c@a.com', '4', '0011'],
                          ['d', 'd@b.com', '1', '0011']])
        t_dist = np.array(3 * [2])
        t_pred = np.array(3 * ['mediocre'])
        assert np.allclose(cfs_dist, t_dist)
        assert np.array_equal(cfs_pred, t_pred)
        assert np.array_equal(np.sort(cfs, axis=0), np.sort(t_cfs, axis=0))

        cfe = ftpc.CounterfactualExplainer(
            model=self.KNN_STRUCT,
            dataset=self.DATASET_STRUCT,
            counterfactual_feature_indices=['q', 'postcode'])
        cfs, cfs_dist, cfs_pred = cfe.explain_instance(self.DATASET_STRUCT[2])
        #
        t_cfs = np.array([('c', 'c@a.com', 6.33, '0011'),
                          ('c', 'c@a.com', 7.33, '1100')],
                         dtype=self.DATASET_STRUCT.dtype)
        t_dist = np.array(2 * [4.78])
        t_pred = np.array(2 * ['mediocre'])
        assert np.array_equal(cfs, t_cfs)
        assert np.allclose(cfs_dist, t_dist)
        assert np.array_equal(cfs_pred, t_pred)

    def test_counterfactuals_manual(self):
        """
        Tests counterfactuals generation with the ``CounterfactualExplainer``.

        The parameters will be set manually.
        """
        user_warning_cf_length = (
            'The value of the max_counterfactual_length parameter is larger '
            'than the number of features. It will be clipped.')
        user_warning_categorical_step = (
            'Step size was provided for one of the categorical features. '
            'Ignoring these ranges.')
        user_warning_cf_out_of_range = (
            'The value ({}) of *{}* feature for this instance is out of the '
            'specified min-max range: {}-{}.')
        user_warning_cf_out_of_values = (
            'The value ({}) of *{}* feature for this instance is out of the '
            'specified values: {}.')

        def email_cat_dist(x, y):
            x_split = x.split('@')
            y_split = y.split('@')
            assert len(x_split) == 2
            assert len(y_split) == 2
            x_split = x_split[1]
            y_split = y_split[1]
            return int(x_split == y_split)

        def email_num_dist(x, y):
            assert int(x) == x
            assert int(y) == y
            x_domain = '_@{}'.format('{0:b}'.format(int(x) & 0b11).zfill(2))
            y_domain = '_@{}'.format('{0:b}'.format(int(y) & 0b11).zfill(2))
            return email_cat_dist(x_domain, y_domain)

        def postcode_cat_dist(x, y):
            assert len(x) == len(y)
            identity_vector = [i[0] != i[1] for i in zip(x, y)]
            return sum(identity_vector) / len(x)

        def postcode_num_dist(x, y):
            assert int(x) == x
            assert int(y) == y
            x_binary = '{0:b}'.format(int(x)).zfill(4)
            y_binary = '{0:b}'.format(int(y)).zfill(4)
            return postcode_cat_dist(x_binary, y_binary)

        str_feature_ranges = {0: ['d'], 1: ['d@b.com', 'e@b.com'], 2: ['4']}
        struct_feature_ranges = {'q': (35, 45), 'postcode': ['0011']}
        feature_steps = {2: 1.75}
        feature_distance_functions_num = {
            1: email_num_dist,
            3: postcode_num_dist
        }
        feature_distance_functions_cat = {
            1: email_cat_dist,
            3: postcode_cat_dist
        }
        feature_distance_functions_mix = {
            'postcode': postcode_cat_dist,
            'email': email_cat_dist
        }

        with pytest.warns(UserWarning) as w:
            cfe = ftpc.CounterfactualExplainer(
                model=self.KNN_NUM,
                dataset=self.DATASET_NUM,
                categorical_indices=[0, 1, 3],
                max_counterfactual_length=5,
                step_sizes=feature_steps,
                distance_functions=feature_distance_functions_num)
        assert len(w) == 1
        assert str(w[0].message) == user_warning_cf_length
        # Outside of categorical range + no counterfactuals
        with pytest.warns(UserWarning) as w:
            cfs, cfs_dist, cfs_pred = cfe.explain_instance(
                np.array([8, 0b0001, 12.57, 0b1100]))
            cfs, cfs_dist, cfs_pred = cfe.explain_instance(
                np.array([8, 0b0001, 12.57, 0b1100]),
                counterfactual_class='non-existing')
        assert len(w) == 2
        rng = ', '.join(['{}.0'.format(i) for i in range(7)])
        rng = '[{}]'.format(rng)
        wmsg = user_warning_cf_out_of_values.format('8.0', 0, rng)
        assert str(w[0].message) == wmsg
        assert str(w[1].message) == wmsg
        #
        assert cfs.size == 0
        assert cfs_dist.size == 0
        assert cfs_pred.size == 0

        with pytest.warns(UserWarning) as w:
            cfe = ftpc.CounterfactualExplainer(
                predictive_function=self.KNN_STR.predict,
                dataset=self.DATASET_STR,
                numerical_indices=[],
                max_counterfactual_length=0,
                step_sizes=feature_steps,
                feature_ranges=str_feature_ranges,
                distance_functions=feature_distance_functions_cat)
        assert len(w) == 1
        assert str(w[0].message) == user_warning_categorical_step
        # No answers
        cfs, cfs_dist, cfs_pred = cfe.explain_instance(
            np.array(['d', 'd@b.com', '4', '0011']),
            counterfactual_class='good')
        #
        assert cfs.size == 0
        assert cfs_dist.size == 0
        assert cfs_pred.size == 0

        cfe = ftpc.CounterfactualExplainer(
            model=self.KNN_STRUCT,
            dataset=self.DATASET_STRUCT,
            categorical_indices=['name', 'email', 'postcode'],
            numerical_indices=['q'],
            counterfactual_feature_indices=['email', 'q', 'postcode'],
            feature_ranges=struct_feature_ranges,
            max_counterfactual_length=1,
            distance_functions=feature_distance_functions_mix)
        # Normalise distance + outside of numerical range
        instance = np.array([('f', 'f@c.com', 45.5, '0011')],
                            dtype=self.DATASET_STRUCT.dtype)[0]
        with pytest.warns(UserWarning) as w:
            cfs, cfs_dist, cfs_pred = cfe.explain_instance(
                instance, normalise_distance=True)
        assert len(w) == 1
        wmsg = user_warning_cf_out_of_range.format('45.5', 'q', 35, 45)
        assert str(w[0].message) == wmsg
        #
        t_cfs = np.array([('f', 'f@c.com', 38, '0011')],
                         dtype=self.DATASET_STRUCT.dtype)
        t_dist = np.array([7.56637298])
        t_pred = np.array(['good'])
        assert np.array_equal(cfs, t_cfs)
        assert np.allclose(cfs_dist, t_dist)
        assert np.array_equal(cfs_pred, t_pred)

        cfe = ftpc.CounterfactualExplainer(
            model=self.KNN_STRUCT,
            categorical_indices=['name', 'email', 'postcode'],
            numerical_indices=['q'],
            counterfactual_feature_indices=['q', 'postcode'],
            feature_ranges=struct_feature_ranges)
        instance = np.array([('f', 'f@c.com', 36.66, '0011')],
                            dtype=self.DATASET_STRUCT.dtype)[0]
        cfs, cfs_dist, cfs_pred = cfe.explain_instance(instance)
        #
        t_cfs = np.array([('f', 'f@c.com', 39, '0011')],
                         dtype=self.DATASET_STRUCT.dtype)
        t_dist = np.array([2.34])
        t_pred = np.array(['mediocre'])
        assert np.array_equal(cfs, t_cfs)
        assert np.allclose(cfs_dist, t_dist)
        assert np.array_equal(cfs_pred, t_pred)

        # Duplicated counterfactual -- filtering is applied
        data = np.array([[0, 0], [1, 1], [0, 1]])
        clf = fum.KNN(k=1)
        clf.fit(data, np.array((['good', 'bad', 'bad'])))
        cfe = ftpc.CounterfactualExplainer(
            model=clf,
            dataset=data,
            max_counterfactual_length=2,
            default_numerical_step_size=0.4)
        cfs, cfs_dist, cfs_pred = cfe.explain_instance(np.array([0, 0]))
        assert np.array_equal(cfs, np.array([[0, 0.8]]))
        assert np.allclose(cfs_dist, np.array([0.8]))
        assert np.array_equal(cfs_pred, np.array(['bad']))

        # Duplicated counterfactual -- filtering is applied
        data = np.array([(0, 0), (1, 1), (0, 1)],
                        dtype=[('a', 'i'), ('b', 'i')])
        clf = fum.KNN(k=1)
        clf.fit(data, np.array((['good', 'bad', 'bad'])))
        cfe = ftpc.CounterfactualExplainer(
            model=clf,
            dataset=data,
            max_counterfactual_length=2,
            default_numerical_step_size=0.4)
        cfs, cfs_dist, cfs_pred = cfe.explain_instance(
            np.array([(0, 0)], dtype=data.dtype)[0])
        assert np.array_equal(
            cfs, np.array([(0, 0.8)], dtype=[('a', float), ('b', float)]))
        assert np.allclose(cfs_dist, np.array([0.8]))
        assert np.array_equal(cfs_pred, np.array(['bad']))
