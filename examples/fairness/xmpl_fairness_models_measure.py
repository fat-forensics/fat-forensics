"""
============================================================
Measuring Fairness of a Predictive Model -- Disparate Impact
============================================================

This example illustrates how to measure Disparate Impact of a predictive model.
In this example we measure the three most common Disparate Impact measures:

* Equal Accuracy;
* Equal Opportunity; and
* Demographic Parity.

.. note::
   Our implementation of the k-nearest neighbours model
   (:class:`fatf.utils.models.models.KNN`) works with structured numpy arrays,
   therefore we do not have to pre-process (e.g., one-hot encode) the
   categorical (stirng-based) features.

    For scikit-learn models all of the categorical features in the data set
    would need to be pre-processed first.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import fatf.utils.data.datasets as fatf_datasets
import fatf.utils.models as fatf_models

import fatf.fairness.models.measures as fatf_mfm

import fatf.utils.metrics.tools as fatf_mt

print(__doc__)

# Load data
hr_data_dict = fatf_datasets.load_health_records()
hr_X = hr_data_dict['data']
hr_y = hr_data_dict['target']
hr_feature_names = hr_data_dict['feature_names']
hr_class_names = hr_data_dict['target_names']

# Train a model
clf = fatf_models.KNN()
clf.fit(hr_X, hr_y)

# Get predictions of the model for the fairness evaluation (which is also the
# training data in this example)
hr_pred = clf.predict(hr_X)

# Select a protected feature
protected_feature = 'gender'

# Get a confusion matrix for all sub-groups according to the split feature
confusion_matrix_per_bin, bin_names = fatf_mt.confusion_matrix_per_subgroup(
    hr_X, hr_y, hr_pred, protected_feature, treat_as_categorical=True)


def print_fairness(metric_name, metric_matrix):
    """Prints out which sub-populations violate a group fairness metric."""
    print('The *{}* group-based fairness metric for *{}* feature split '
          'are:'.format(metric_name, protected_feature))
    for grouping_i, grouping_name_i in enumerate(bin_names):
        j_offset = grouping_i + 1
        for grouping_j, grouping_name_j in enumerate(bin_names[j_offset:]):
            grouping_j += j_offset
            is_not = ' >not<' if metric_matrix[grouping_i, grouping_j] else ''

            print('    * The fairness metric is{} satisfied for "{}" and "{}" '
                  'sub-populations.'.format(is_not, grouping_name_i,
                                            grouping_name_j))


##############################################################################
# Equal Accuracy
# --------------
#
# First, let's measure whether the model is fair according to the Equal
# Accuracy metric.

# Get the Equal Accuracy binary matrix
equal_accuracy_matrix = fatf_mfm.equal_accuracy(confusion_matrix_per_bin)

# Print out fairness
print_fairness('Equal Accuracy', equal_accuracy_matrix)

##############################################################################
# Equal Opportunity
# -----------------
#
# Now, let's see whether the model is fair according to the Equal Opportunity
# metric.

# Get the Equal Opportunity binary matrix
equal_opportunity_matrix = fatf_mfm.equal_opportunity(confusion_matrix_per_bin)

# Print out fairness
print_fairness('Equal Opportunity', equal_opportunity_matrix)

##############################################################################
# Demographic Parity
# ------------------
#
# Finally, let's measure the Demographic Parity of the model.

# Get the Demographic Parity binary matrix
demographic_parity_matrix = fatf_mfm.demographic_parity(
    confusion_matrix_per_bin)

# Print out fairness
print_fairness('Demographic Parity', demographic_parity_matrix)

##############################################################################
# ----
#
# Based on these results we can easily see that **Demographic Parity** is the
# only fairness metric that is violated.
#
# ----
