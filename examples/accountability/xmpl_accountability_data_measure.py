"""
===================================================
Measuring Robustness of a Data Set -- Sampling Bias
===================================================

This example illustrates how to identify Sampling Bias for a data set grouping
for a selected feature.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import fatf.utils.data.datasets as fatf_datasets

import fatf.accountability.data.measures as fatf_dam

print(__doc__)

# Load data
iris_data_dict = fatf_datasets.load_iris()
iris_X = iris_data_dict['data']
iris_y = iris_data_dict['target'].astype(int)
iris_feature_names = iris_data_dict['feature_names']
iris_class_names = iris_data_dict['target_names']

# Select a feature for which the Sampling Bias be measured
selected_feature_index = 2
selected_feature_name = iris_feature_names[selected_feature_index]

# Define grouping on the selected feature
selected_feature_grouping = [2.5, 4.75]

# Get counts, weights and names of the specified grouping
grp_counts, grp_weights, grp_names = fatf_dam.sampling_bias(
    iris_X, selected_feature_index, selected_feature_grouping)

# Print out counts per group
print('The counts for groups defined on "{}" feature (index {}) are:'
      ''.format(selected_feature_name, selected_feature_index))
for g_name, g_count in zip(grp_names, grp_counts):
    is_are = 'is' if g_count == 1 else 'are'
    print('    * For the population split *{}* there {}: '
          '{} data points.'.format(g_name, is_are, g_count))

# Get the disparity grid
bias_grid = fatf_dam.sampling_bias_grid_check(grp_counts)

# Print out disparity per every grouping pair
print('\nThe Sampling Bias for *{}* feature (index {}) grouping is:'
      ''.format(selected_feature_name, selected_feature_index))
for grouping_i, grouping_name_i in enumerate(grp_names):
    j_offset = grouping_i + 1
    for grouping_j, grouping_name_j in enumerate(grp_names[j_offset:]):
        grouping_j += j_offset
        is_not = '' if bias_grid[grouping_i, grouping_j] else ' no'

        print('    * For "{}" and "{}" groupings there >is{}< Sampling Bias.'
              ''.format(grouping_name_i, grouping_name_j, is_not))
