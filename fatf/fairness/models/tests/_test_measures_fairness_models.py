"""
Tests methods for fairness checks.
"""
# Author: Rafael Poyiadzi <rp13102@bristol.ac.uk>
#         Kacper Sokol <k.sokol@bristol.ac.uk>
# License: BSD 3 clause

import math
import pytest

import numpy as np

from fatf.fairness.metrics import _get_confusion_matrix, filter_dataset, split_dataset, get_cross_product
from fatf.fairness.metrics import get_mask, get_weights_costsensitivelearning, get_counts, apply_combination_filter

from metrics import FairnessChecks
from sklearn.linear_model import LogisticRegression


input0 = [0, 0, 0, 1, 1, 1]
input1 = [1, 1, 0, 1, 0, 1]

labels0 = [1, 0]
output0 = [[2, 1],
           [2, 1]]

labels1 = [0, 1]
output1 = [[1, 2],
           [1, 2]]

@pytest.mark.parametrize("input0, input1, labels, expected_output",
                         [(input0, input1, labels0, output0),
                          (input0, input1, labels1, output1)])
def test_get_confusion_matrix(input0, input1, labels, expected_output):
    output_list = [item for sublist in expected_output for item in sublist]
    output_cm = _get_confusion_matrix(input0, input1, labels)
    output_cm_list = [item for sublist in output_cm.tolist() for item in sublist]
    assert np.all([output_list[i] == output_cm_list[i] for i in range(len(output_list))])

x0_input = np.array([(0, 0, 1),
                 (1, 1, 0),
                 (0, 1, 1),
                 (0, 1, 0),],
      dtype=[('feature1', '<i4'), ('feature2', '<i4'), ('gender', '<i4')])

x1_input = np.array([(0, 0, 1),
                 (1, 1, 1),
                 (0, 1, 1),
                 (0, 1, 1),],
      dtype=[('feature1', '<i4'), ('feature2', '<i4'), ('gender', '<i4')])

x2_input = np.array([(0, 0, 0),
                 (1, 1, 0),
                 (0, 1, 0),
                 (0, 1, 0),],
      dtype=[('feature1', '<i4'), ('feature2', '<i4'), ('gender', '<i4')])

targets = np.array([1, 1, 0, 0])
predictions = np.array([0, 1, 0, 1])

input_data0 = [x0_input, targets, predictions]
input_data1 = [x1_input, targets, predictions]
input_data2 = [x2_input, targets, predictions]

x0_output = np.array([(0, 0, 1),
                      (0, 1, 1)],
      dtype=[('feature1', '<i4'), ('feature2', '<i4'), ('gender', '<i4')])

targets0_output = np.array([1, 0])
predictions0_output = np.array([0, 0])

x2_output = np.array([],
      dtype=[('feature1', '<i4'), ('feature2', '<i4'), ('gender', '<i4')])

expected_data0 = [x0_output, targets0_output, predictions0_output]
expected_data1 = [x1_input, targets, predictions]
expected_data2 = [x2_output, np.array([], dtype='int32'), np.array([], dtype='int32')]

@pytest.mark.parametrize("input_data, feature, feature_value, expected_data",
                         [(input_data0, 'gender', 1, expected_data0),
                          (input_data1, 'gender', 1, expected_data1),
                          (input_data2, 'gender', 1, expected_data2)])
def test_filter_dataset(input_data, feature, feature_value, expected_data):
    X, targets, predictions = input_data
    x_expected, targets_expected, predictions_expected = expected_data

    x_output, targets_output, predictions_output = \
        filter_dataset(X, targets, predictions, feature, feature_value)
    assert np.all(x_expected == x_output)
    assert np.all(targets_expected == targets_output)
    assert np.all(predictions_expected == predictions_output)

expected_splits0 = [(0,
                     (np.array([(1, 1, 0),
                                (0, 1, 0)],
                          dtype=[('feature1', '<i4'), ('feature2', '<i4'), ('gender', '<i4')]),
                    np.array([1, 0]),
                    np.array([1, 1]))
                      ),
                    (1,
                     (x0_output,
                    np.array([1, 0]),
                    np.array([0, 0])
                      )
                     )
                    ]

expected_splits1 = [(0,
                     (np.array([],
                          dtype=[('feature1', '<i4'), ('feature2', '<i4'), ('gender', '<i4')]),
                    np.array([]),
                    np.array([]))
                      ),
                    (1,
                     (x1_input,
                      targets,
                      predictions
                      )
                     )
                    ]

expected_splits2 = [(0,
                     (x2_input,
                      targets,
                      predictions
                      )
                     ),
                    (1,
                     (np.array([],
                          dtype=[('feature1', '<i4'), ('feature2', '<i4'), ('gender', '<i4')]),
                    np.array([]),
                    np.array([]))
                      )
                    ]
@pytest.mark.parametrize("input_data, feature, expected_splits",
                         [(input_data0, 'gender', expected_splits0),
                          (input_data1, 'gender', expected_splits1),
                          (input_data2, 'gender', expected_splits2)])
def test_split_dataset(input_data, feature, expected_splits):
    X, targets, predictions = input_data
    labels = [0, 1]
    output_splits = split_dataset(X, targets, predictions, feature, labels)
    for idx in range(2):
        x_expected, targets_expected, predictions_expected = expected_splits[idx][1]
        x_output, targets_output, predictions_output = output_splits[idx][1]

        assert expected_splits[idx][0] == output_splits[idx][0]
        assert np.all(x_expected == x_output)
        assert np.all(targets_expected == targets_output)
        assert np.all(predictions_expected == predictions_output)

input_data_crossproduct = np.array([('a', 'A', '0'),
                        ('b', 'B', '1')],
                        dtype=[('first', '<U1'),
                               ('second', '<U1'),
                               ('third', '<U1')]
                        )

input_data_crossproduct1 = np.array([('a', 'A', 0, 20),
                               ('b', 'B', 1, 30),
                               ('a', 'B', 0, 40),
                               ('b', 'B', 1, 50),
                               ('a', 'A', 0, 60),
                               ('b', 'B', 1, 70)],
                            dtype=[('first', '<U1'),
                                   ('second', '<U1'),
                                   ('third', '<i4'),
                                   ('age', '<i4')]
                            )

expected_crossproduct0 = [('a',), ('b',)]
expected_crossproduct1 = [('a', 'A',), ('a', 'B',), ('b', 'A',), ('b', 'B',)]

INF = math.inf
expected_crossproduct2 = [((-INF, 25), ), ((25, 50), ), ((50, INF), )]
expected_crossproduct3 = [('a', (-INF, 45), ), ('a', (45, INF), ),
                          ('b', (-INF, 45), ), ('b', (45, INF), )]
@pytest.mark.parametrize("input_data, features_to_check, expected_output, boundaries",
                         [(input_data_crossproduct, ['first'], expected_crossproduct0, {}),
                          (input_data_crossproduct, ['first', 'second'], expected_crossproduct1, {}),
                          (input_data_crossproduct1, ['age'], expected_crossproduct2, {'age': [25, 50]}),
                          (input_data_crossproduct1, ['first', 'age'], expected_crossproduct3, {'age': [45]})])
def test_get_cross_product(input_data, features_to_check, expected_output, boundaries):
    output = get_cross_product(input_data, features_to_check, boundaries)
    assert set(output) == set(expected_output)

input_data_mask0 = np.array([('a', 'A', 0, 20),
                               ('b', 'B', 1, 30),
                               ('a', 'B', 0, 40),
                               ('b', 'B', 1, 50),
                               ('a', 'A', 0, 60),
                               ('b', 'B', 1, 70)],
                            dtype=[('first', '<U1'),
                                   ('second', '<U1'),
                                   ('third', '<i4'),
                                   ('age', '<i4')]
                            )
expected_output_mask0 = [True, False, True, False, True, False]
expected_output_mask1 = [True, False, False, False, True, False]
expected_output_mask2 = [True, False, False, False, False, False]
expected_output_mask3 = [True, False, True, False, False, False]
@pytest.mark.parametrize("input_dataset, features_to_check, combination, expected_output, boundaries_for_numerical",
                         [(input_data_mask0, ['first'], ('a',), expected_output_mask0, {}),
                          (input_data_mask0, ['first', 'second'], ('a', 'A', ), expected_output_mask1, {}),
                          (input_data_mask0, ['age'], ([15, 25], ), expected_output_mask2, {'age':[]}),
                          (input_data_mask0, ['first', 'age'], ('a', [15, 45], ), expected_output_mask3, {'age':[]})])
def test_get_mask(input_dataset, features_to_check, combination, expected_output, boundaries_for_numerical):
    output = get_mask(input_dataset, features_to_check, combination, boundaries_for_numerical)
    assert set(output) == set(expected_output)

expected_weights0 = np.array([1, 1, 1, 1, 1, 1], dtype=float).reshape(-1, 1)
expected_weights1 = np.array([2, 1, 1, 1, 2, 1], dtype=float).reshape(-1, 1)
expected_weights2 = np.array([1.5, 1, 3, 1, 1.5, 1], dtype=float).reshape(-1, 1)

@pytest.mark.parametrize("input_dataset, features_to_check, expected_weights",
                         [(input_data_mask0, ['first'], expected_weights0),
                          (input_data_mask0, ['second'], expected_weights1),
                          (input_data_mask0, ['first', 'second'], expected_weights2)])
def test_get_weights_costsensitivelearning(input_dataset, features_to_check, expected_weights):
    target_field = 'third'
    cross_product = get_cross_product(input_dataset, features_to_check, {})
    counts = get_counts(input_dataset, target_field, features_to_check, cross_product, {})
    output_weights = get_weights_costsensitivelearning(input_dataset, features_to_check, counts, {})
    assert np.all(output_weights == expected_weights)

input_data_counts = np.array([('a', 'A', 0, 20),
                               ('b', 'B', 1, 30),
                               ('a', 'B', 0, 40),
                               ('b', 'B', 1, 50),
                               ('a', 'A', 0, 60),
                               ('b', 'B', 0, 70)],
                            dtype=[('first', '<U1'),
                                   ('second', '<U1'),
                                   ('third', '<i4'),
                                   ('age', '<i4')]
                            )

expected_counts0 = {('a',): {0: 3},
                    ('b',): {0: 1, 1: 2}}

expected_counts1 = {('a', 'A', ): {0: 2},
                    ('a', 'B', ): {0: 1},
                    ('b', 'B', ): {0: 1, 1: 2}
                    }

@pytest.mark.parametrize("input_dataset, features_to_check, expected_counts",
                         [(input_data_counts, ['first'], expected_counts0),
                          (input_data_counts, ['first', 'second'], expected_counts1)])
def test_get_counts(input_dataset, features_to_check, expected_counts):
    target_field = 'third'
    cross_product = get_cross_product(input_dataset, features_to_check)
    output_counts = get_counts(input_dataset, target_field, features_to_check, cross_product, {})
    for key, val in output_counts.items():
        assert dict(output_counts[key]) == expected_counts[key]

input_data_apply = np.array([('a', 'A', 0, 1),
                               ('b', 'B', 1, 1),
                               ('a', 'B', 0, 0),
                               ('b', 'B', 1, 0),
                               ('a', 'A', 0, 0),
                               ('b', 'B', 0, 1)],
                            dtype=[('first', '<U1'),
                                   ('second', '<U1'),
                                   ('third', '<i4'),
                                   ('fourth', '<i4')]
                            )

expected_apply0 = ([1, 0, 0],
                   [0, 0, 0])

expected_apply1 = ([1, 0],
                   [0, 0])
@pytest.mark.parametrize("input_dataset, features_to_check, combination, expected_output",
                         [(input_data_apply, ['first'], ('a', ), expected_apply0),
                          (input_data_apply, ['first', 'second'], ('a', 'A', ), expected_apply1)])
def test_apply_combination_filter(input_dataset, features_to_check, combination, expected_output):
    target_field = 'third'
    prediction_field = 'fourth'
    output = apply_combination_filter(input_dataset, prediction_field, target_field, features_to_check, combination)
    assert np.all(output[0] == expected_output[0])
    assert np.all(output[1] == expected_output[1])

def _test_testing():
    def remove_field(dataset, field):
        field_names = list(dataset.dtype.names)
        if field in field_names:
            field_names.remove(field)
        return dataset[field_names]

    testdata3 = np.array([
            ('Heidi Mitchell', 'uboyd@hotmail.com', 74, 52, 0, '0011', 1, '03/06/2018', 0),
           ('Tina Burns', 'stevenwheeler@williams.bi',  3, 86, 1, '0011', 0, '26/09/2017', 1),
           ('Justin Brown', 'velasquezjake@gmail.com', 3, 86, 0, '0011', 1, '31/12/2015', 0),
           ('Brent Parker', 'kennethsingh@strong-foley', 70, 57, 0, '0011', 1, '02/10/2011', 0),
           ('Bryan Norton', 'erica36@hotmail.com', 48, 57, 0, '1100', 0, '09/09/2012', 1),
           ('Ms. Erin Craig', 'ritterluke@gmail.com', 30, 98, 0, '1100', 0, '04/11/2006', 1),
           ('Gerald Park', 'larrylee@hayes-brown.net', 41, 73, 1, '1100', 0, '15/12/2015', 0),
           ],
          dtype=[('name', '<U16'), ('email', '<U25'), ('age', '<i4'),
                 ('weight', '<i4'), ('gender', '<i4'), ('zipcode', '<U6'),
                 ('target', '<i4'), ('dob', '<U10'), ('prediction', '<i4')])

    def create_dataset():
        list_of_dictionaries = get_data()

        desired_keys = ['name',
                        'data',
                        'treatment',
                        'distance_func'
                        ]

        dts = []
        treatments = {
                    'Protected': [],
                    'Feature': [],
                    'ToIgnore': [],
                    'Target': []
                    }
        distance_funcs = {}
        data = []


        for dictionary in list_of_dictionaries:
            current_dictionary_keys = dictionary.keys()
            for key in desired_keys:
                if key not in current_dictionary_keys:
                    raise ValueError('One of the provided dictionaries does not have the key: ' + str(key))

            field_name = dictionary['name']
            field_col = dictionary['data']
            if type(field_col) != np.ndarray:
                raise TypeError(str(field_name) + ' data should be of type numpy.ndarray.')

            data.append(field_col)

            dts.append((field_name, field_col.dtype))
            distance_funcs[field_name] = dictionary['distance_func']

            field_treatment = dictionary['treatment']

            if field_treatment == 'Protected':
                treatments['Protected'].append(field_name)
            elif field_treatment == 'Feature':
                treatments['Feature'].append(field_name)
            elif field_treatment == 'Target':
                treatments['Target'].append(field_name)
            elif field_treatment == 'ToIgnore':
                treatments['ToIgnore'].append(field_name)
            else:
                raise ValueError('Unknown treatment')

        N = data[0].shape[0]
        if not np.all(column.shape[0] == N for column in data):
            raise ValueError('Data provided is of different length.')

        dataset = np.array([item for item in zip(*data)], dtype=dts)
        return dataset, treatments, distance_funcs

    def get_dictionary(field_name, field_data, field_treatment, field_distance_func):
        dictionary =  {
                    'name': field_name,
                    'data': field_data,
                    'treatment': field_treatment,
                    'distance_func': field_distance_func
                    }

        return dictionary

    def zipcode_dist(x, y):
        n = len(x)
        t = sum([item[0] == item[1] for item in zip(x, y)])
        return 1 - t/n

    def get_data():
        age_dict = get_dictionary(field_name = 'Age',
                                  field_data = testdata3['age'],
                                  field_treatment = 'Feature',
                                  field_distance_func = lambda x, y: abs(x - y)
                                  )

        weight_dict = get_dictionary(field_name = 'Weight',
                                  field_data = testdata3['weight'],
                                  field_treatment = 'Feature',
                                  field_distance_func = lambda x, y: abs(x - y)
                                  )

        disease_dict = get_dictionary(field_name = 'Target',
                                  field_data = testdata3['target'],
                                  field_treatment = 'Target',
                                  field_distance_func = lambda x, y: x == y
                                  )



        zipcode_dict = get_dictionary(field_name = 'Zipcode',
                                  field_data = testdata3['zipcode'],
                                  field_treatment = 'Feature',
                                  field_distance_func = zipcode_dist
                                  )

        gender_dict = get_dictionary(field_name = 'Gender',
                                  field_data = testdata3['gender'],
                                  field_treatment = 'Protected',
                                  field_distance_func = lambda x, y: x == y
                                  )

        prediction_dict = get_dictionary(field_name = 'Prediction',
                                  field_data = testdata3['prediction'],
                                  field_treatment = 'ToIgnore',
                                  field_distance_func = None
                                  )

        return [age_dict, weight_dict, disease_dict, zipcode_dict, gender_dict, prediction_dict]

    def get_boundaries(field, increments=5):
        max_val = np.max(field)
        min_val = np.min(field)
        return np.linspace(min_val, max_val+1, increments)

    dataset, treatments, distance_funcs = create_dataset()
    targets = np.array(dataset['Target'])
    predictions = dataset['Prediction']
    dataset = remove_field(dataset, 'Target')
    dataset = remove_field(dataset, 'Prediction')

    features_to_check = ['Gender']

    mdl = FairnessChecks(dataset,
                         targets,
                         distance_funcs,
                         protected = treatments['Protected'][0],
                         toignore = treatments['ToIgnore']
                         )

    c=mdl.check_systemic_bias()


    d=mdl.check_sampling_bias(features_to_check=features_to_check)


    features_to_check = ['Age', 'Gender']
    boundaries = {'Age': []}
    for key, value in boundaries.items():
        if len(value) == 0:
            boundaries[key] = get_boundaries(dataset[key])

    e=mdl.check_sampling_bias(features_to_check=features_to_check,
                              return_weights = True,
                              boundaries_for_numerical = boundaries)

    f=mdl.check_systematic_error(predictions = predictions,
                                 features_to_check=['Gender'],
                                 boundaries_for_numerical=boundaries)


    aggregated_checks = mdl.perform_checks_on_split(
                                                    get_summary = True,
                                                    conditioned_field='Zipcode',
                                                    condition='0011')

    aggregated_checks2 = mdl.perform_checks_on_split(
                                                    get_summary = True,
                                                    requested_checks=['accuracy'])

    X = remove_field(dataset, 'Zipcode')

    model = LogisticRegression()

    newx = np.array(X.tolist())
    model.fit(newx, targets)


    cm = mdl.counterfactual_fairness(model, 'Gender', X, [0, 1, 2])


    g = mdl.individual_fairness(model, newx)


    #####################################
    ######################################
    ####################################

    # =============================================================================
    # dataset_ = dataset.copy(order = 'K')
    # dataset_['Zipcode'][:4] = 0
    # dataset_['Zipcode'][4:] = 1
    # features_to_check = [3]
    # dataset_=dataset_.copy(order='K').astype(dtype=[(name, '<f4') for name in dataset.dtype.names]).view('<f4').reshape(dataset.shape + (-1,))
    #
    #
    # mdl = FairnessChecks(dataset_,
    #                      targets,
    #                      distance_funcs,
    #                      protected = 3,#treatments['Protected'][0],
    #                      toignore = treatments['ToIgnore']
    #                      )
    #
    # c2=mdl.check_systemic_bias()
    #
    #
    # d2=mdl.check_sampling_bias(features_to_check=features_to_check)
    #
    #
    # features_to_check = [0, 3]
    # boundaries = {0: []}
    # for key, value in boundaries.items():
    #     if len(value) == 0:
    #         boundaries[key] = get_boundaries(dataset_[:, key])
    #
    # e2=mdl.check_sampling_bias(features_to_check=features_to_check,
    #                           return_weights = True,
    #                           boundaries_for_numerical = boundaries)
    #
    # f2=mdl.check_systematic_error(predictions = predictions,
    #                              features_to_check=[3],
    #                              requested_checks='all',
    #                              boundaries_for_numerical=boundaries)
    #
    #
    # aggregated_checks2b = mdl.perform_checks_on_split(
    #                                                 get_summary = True,
    #                                                 requested_checks=['accuracy'])
    #
    #
    # aggregated_checksb = mdl.perform_checks_on_split(
    #                                                 get_summary = True,
    #                                                 requested_checks=['accuracy'],
    #                                                 conditioned_field=2,
    #                                                 condition=0)
    #
    # X = dataset_
    # model = LogisticRegression()
    #
    # newx = np.array(X.tolist())
    # model.fit(newx, targets)
    #
    #
    # cm = mdl.counterfactual_fairness(model, 3, X, [0, 1])
    #
    #
    # g = mdl.individual_fairness(model, newx)
    #
    #
    # =============================================================================
