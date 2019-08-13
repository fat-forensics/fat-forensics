"""
=========================================================================
Measuring Robustness of a Predictive Model -- Systematic Performance Bias
=========================================================================

This example illustrates how to measure Systematic Performance Bias based on a
selected predictive performance metric (*accuracy* in this example).
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import fatf.utils.data.datasets as fatf_datasets
import fatf.utils.models as fatf_models

import fatf.accountability.models.measures as fatf_mam

import fatf.utils.metrics.subgroup_metrics as fatf_smt

print(__doc__)

# Load data
iris_data_dict = fatf_datasets.load_iris()
iris_X = iris_data_dict['data']
iris_y = iris_data_dict['target'].astype(int)
iris_feature_names = iris_data_dict['feature_names']
iris_class_names = iris_data_dict['target_names']

# Train a model
clf = fatf_models.KNN()
clf.fit(iris_X, iris_y)

# Get predictions of the model for the fairness evaluation (which is also the
# training data in this example)
iris_pred = clf.predict(iris_X)

# Select a predictive performance metric
predictive_performance_metric = 'accuracy'

# Select a feature for which the difference in performance should be measured
selected_feature_index = 1
selected_feature_name = iris_feature_names[selected_feature_index]

# Define grouping on the selected feature
selected_feature_grouping = [3]

# Compute accuracy per group in the feature
population_metrics, population_names = fatf_smt.performance_per_subgroup(
    iris_X,
    iris_y,
    iris_pred,
    selected_feature_index,
    groupings=selected_feature_grouping,
    metric=predictive_performance_metric)

# Print out performance per grouping
print('The *{}* for groups defined on "{}" feature (feature index '
      '{}):'.format(predictive_performance_metric, selected_feature_name,
                    selected_feature_index))
for p_name, p_metric in zip(population_names, population_metrics):
    print('    * For the population split *{}* the {} is: '
          '{:.2f}.'.format(p_name, predictive_performance_metric, p_metric))

# Evaluate Systematic Performance Bias
bias_grid = fatf_mam.systematic_performance_bias_grid(population_metrics)

# Print out Systematic Performance Bias for each grouping pair
print('\nThe *{}-based* Systematic Performance Bias for *{}* feature (index '
      '{}) grouping is:'.format(predictive_performance_metric,
                                selected_feature_name, selected_feature_index))
for grouping_i, grouping_name_i in enumerate(population_names):
    j_offset = grouping_i + 1
    for grouping_j, grouping_name_j in enumerate(population_names[j_offset:]):
        grouping_j += j_offset
        is_not = '' if bias_grid[grouping_i, grouping_j] else ' no'

        print('    * For "{}" and "{}" groupings there >is{}< Systematic '
              'Performance Bias.'.format(grouping_name_i, grouping_name_j,
                                         is_not))
