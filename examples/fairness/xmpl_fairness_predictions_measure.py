"""
=============================================================
Measuring Fairness of a Prediction -- Counterfactual Fairness
=============================================================

This example illustrates how scrutinise a data point under the counterfactual
fairness assumption.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import numpy as np

import fatf.utils.data.datasets as fatf_datasets
import fatf.utils.models as fatf_models

import fatf.fairness.predictions.measures as fatf_pfm

import fatf.transparency.predictions.counterfactuals as fatf_cf

print(__doc__)

# Load data
hr_data_dict = fatf_datasets.load_health_records()
hr_X = hr_data_dict['data']
hr_y = hr_data_dict['target']
hr_feature_names = hr_data_dict['feature_names']
hr_class_names = hr_data_dict['target_names']

# Map target indices to target names
hr_y = np.array([hr_class_names[i] for i in hr_y])

# Drop the unique identifiers (features)
unique_identifiers = ['name', 'email', 'zipcode', 'dob']
columns_to_keep = [i for i in hr_X.dtype.names if i not in unique_identifiers]
#
hr_X = hr_X[columns_to_keep]
hr_feature_names = [i for i in hr_feature_names if i not in unique_identifiers]

# Train a model
clf = fatf_models.KNN()
clf.fit(hr_X, hr_y)

# Select a data point to evaluate its counterfactual fairness
data_point_index = 4 + 2
data_point = hr_X[data_point_index]
data_point_y = hr_y[data_point_index]

# Select a set of protected features
protected_features = ['gender', 'age']

# Print out the protected features
assert protected_features, 'The protected features list cannot be empty.'
person = ' is' if len(protected_features) == 1 else 's are'
print('The following fautre{} considered protected:'.format(person))
for feature_name in protected_features:
    print('    * "{}".'.format(feature_name))

# Print the instance
print('\nEvaluating counterfactual fairness of a data point (index {}) of '
      'class *{}* with the following features:'.format(data_point_index,
                                                       data_point_y))
for feature_name in data_point.dtype.names:
    print('    * The feature *{}* has value: {}.'.format(
        feature_name, data_point[feature_name]))

# Compute counterfactually unfair examples
cfs, cfs_distances, cfs_classes = fatf_pfm.counterfactual_fairness(
    instance=data_point,
    protected_feature_indices=protected_features,
    model=clf,
    default_numerical_step_size=1,
    dataset=hr_X)

# Textualise possible counterfactually unfair data points
cfs_text = fatf_cf.textualise_counterfactuals(
    data_point,
    cfs,
    instance_class=data_point_y,
    counterfactuals_distances=cfs_distances,
    counterfactuals_predictions=cfs_classes)
print('\n{}'.format(cfs_text))
