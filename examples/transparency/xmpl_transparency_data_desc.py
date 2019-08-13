"""
================================
Using Data Description Explainer
================================

This example illustrates how to use the Data Description to interpret a data
set. (See the :mod:`fatf.transparency.data.describe_functions` module for more
details.)
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

from pprint import pprint
import numpy as np

import fatf.utils.data.datasets as fatf_datasets

import fatf.transparency.data.describe_functions as fatf_dd

print(__doc__)

# Load data
iris_data_dict = fatf_datasets.load_iris()
iris_X = iris_data_dict['data']
iris_y = iris_data_dict['target'].astype(int)
iris_feature_names = iris_data_dict['feature_names']
iris_class_names = iris_data_dict['target_names']

###############################################################################
# Start by describing all of the features in the data set.

# Explain all of the features
features_description = fatf_dd.describe_array(iris_X)

# Conver feature ids into feature names
named_features_description = dict()
for fdi in features_description.items():
    feature_id, feature_description = fdi
    feature_name = iris_feature_names[feature_id]

    named_features_description[feature_name] = feature_description

print('Data Description for each feature:')
pprint(named_features_description)

###############################################################################
# Now describe the 'petal width (cm)' per class.

# Select the 'petal width (cm)' feature
selected_feature_id = 3
selected_feature_name = iris_feature_names[selected_feature_id]

# Group the data points per class
per_class_row_mask = dict()
for class_index, class_name in enumerate(iris_class_names):
    per_class_row_mask[class_name] = iris_y == class_index

# Explain the 'petal width (cm)' feature per class
per_class_explanation = dict()
for class_name, class_mask in per_class_row_mask.items():
    class_array = iris_X[class_mask, selected_feature_id]
    per_class_explanation[class_name] = fatf_dd.describe_array(class_array)

print('Per-class Data Description of each feature for class '
      "'{}' (class index {}):".format(selected_feature_name,
                                      selected_feature_id))
pprint(per_class_explanation)

###############################################################################
# Finally, describe the class distribution.

# Get the Data Description for the target variable
target_explanation = fatf_dd.describe_categorical_array(iris_y)

print('Data Description of the target array:')
pprint(target_explanation)

# Since the targer array is numerical, we can convert it to class names first
iris_y_named = np.array([iris_class_names[i] for i in iris_y])
target_explanation_named = fatf_dd.describe_categorical_array(iris_y_named)

print('Data Description of the target array mapped to class names:')
pprint(target_explanation_named)
