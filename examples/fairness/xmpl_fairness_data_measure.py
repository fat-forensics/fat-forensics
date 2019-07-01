"""
================================
Measuring Fairness of a Data Set
================================

This example illustrates how to find unfair rows in a data set using the
:func:`fatf.fairness.data.measures.systemic_bias` function and how to check
whether each class is distributed equally between values of a selected feature,
i.e. measuring the Sample Size Disparity (with :func:`fatf.utils.data.tools.\
group_by_column` function).

.. note::
   Please note that this example uses a data set that is represented as a
   structured numpy array, which supports mixed data types among columns with
   the features (columns) being index by the feature name rather than by
   consecutive integers.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

from pprint import pprint
import numpy as np

import fatf.utils.data.datasets as fatf_datasets

import fatf.fairness.data.measures as fatf_dfm

import fatf.utils.data.tools as fatf_data_tools

print(__doc__)

# Load data
hr_data_dict = fatf_datasets.load_health_records()
hr_X = hr_data_dict['data']
hr_y = hr_data_dict['target']
hr_feature_names = hr_data_dict['feature_names']
hr_class_names = hr_data_dict['target_names']

#################################################
# Systemic Bias
# -------------
#
# Before we proceed, we need to select which feature are **protected**, i.e.
# which ones are illegal to use when generating the prediction.
#
# We use them to see whether the data set contains rows that differ in some of
# the protected features and the labels (ground truth) but not in the rest of
# the features.
#
# The example presented below is rather naive as we do not have access to a
# more complicated dataset within the FAT-Forensics package. To demonstrate the
# functionality of the we indicate all but one feature to be protected, hence
# we are guaranteed to find quite a few unfair rows in the health records data
# set. This means that "unfair" data rows are the ones that have the same value
# of the *diagnosis* feature (with rest of the feature values being
# unimportant) and differ in their target (ground truth) value.
#
# Systematic bias is expressed here as a square matrix (numpy array) of length
# equal to the number of rows in the data array. Each element of this matrix
# is a boolean indicating whether the rows in the data array with a particular
# pair of indices (the row and column indices of the boolean matrix) violate
# the aforementioned fairness criterion.

# Select which features should be treated as protected
protected_features = [
    'name', 'email', 'age', 'weight', 'gender', 'zipcode', 'dob'
]

# Compute the data fairness matrix
data_fairness_matrix = fatf_dfm.systemic_bias(hr_X, hr_y, protected_features)

# Check if the data set is unfair (at least one unfair pair of data points)
is_data_unfair = fatf_dfm.systemic_bias_check(data_fairness_matrix)

# Identify which pairs of indices cause the unfairness
unfair_pairs_tuple = np.where(data_fairness_matrix)
unfair_pairs = []
for i, j in zip(*unfair_pairs_tuple):
    pair_a, pair_b = (i, j), (j, i)
    if pair_a not in unfair_pairs and pair_b not in unfair_pairs:
        unfair_pairs.append(pair_a)

# Print out whether the fairness condition is violated
if is_data_unfair:
    unfair_n = len(unfair_pairs)
    unfair_fill = ('is', '') if unfair_n == 1 else ('are', 's')
    print('\nThere {} {} pair{} of data points that violates the fairness '
          'criterion.\n'.format(unfair_fill[0], unfair_n, unfair_fill[1]))
else:
    print('The data set is fair.\n')

# Show the first pair of violating rows
pprint(hr_X[[unfair_pairs[0][0], unfair_pairs[0][1]]])

###############################################################################
# Sample Size Disparity
# ---------------------
#
# The measure of Sample Size Disparity can be achieved by calling the
# :func:`fatf.utils.data.tools.group_by_column` grouping function and counting
# the number of instances in each group. By doing that for the *target vector*
# (ground truth) we can see whether the classes in our data set are balanced
# for each sub-group defined by a specified set of values for that feature.
#
# In the example below we will check whether there are roughly the same number
# of data points collected for *males* and *females*. Then we will see whether
# the class distribution (*fail* and *success*) for these two sub-populations
# is similar.

# Group the data based on the unique values of the 'gender' column
grouping_column = 'gender'
grouping_indices, grouping_names = fatf_data_tools.group_by_column(
    hr_X, grouping_column, treat_as_categorical=True)

# Print out the data distribution for the grouping
print('The grouping based on the *{}* feature has the '
      'following distribution:'.format(grouping_column))
for grouping_name, grouping_idx in zip(grouping_names, grouping_indices):
    print('    * "{}" grouping has {} instances.'.format(
        grouping_name, len(grouping_idx)))

# Get the class distribution for each sub-grouping
grouping_class_distribution = dict()
for grouping_name, grouping_idx in zip(grouping_names, grouping_indices):
    sg_y = hr_y[grouping_idx]
    sg_classes, sg_counts = np.unique(sg_y, return_counts=True)

    grouping_class_distribution[grouping_name] = dict()
    for sg_class, sg_count in zip(sg_classes, sg_counts):
        sg_class_name = hr_class_names[sg_class]

        grouping_class_distribution[grouping_name][sg_class_name] = sg_count

# Print out the class distribution per sub-group
print('\nThe class distribution per sub-population:')
for grouping_name, class_distribution in grouping_class_distribution.items():
    print('    * For the "{}" grouping the classes are distributed as '
          'follows:'.format(grouping_name))
    for class_name, class_count in class_distribution.items():
        print('        - The class *{}* has {} data points.'.format(
            class_name, class_count))
