"""
=========================================
Using Counterfactual Prediction Explainer
=========================================

This example illustrates how to use the Counterfactual Prediction explainer
(:class:`fatf.transparency.predictions.counterfactuals.\
CounterfactualExplainer`) and how to interpret the 3-tuple that it returns by
"textualising" it (:func:`fatf.transparency.predictions.counterfactuals.\
textualise_counterfactuals`).
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

from pprint import pprint
import numpy as np

import fatf.utils.data.datasets as fatf_datasets
import fatf.utils.models as fatf_models

import fatf.transparency.predictions.counterfactuals as fatf_cf

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

# Create a Counterfactual Explainer
cf_explainer = fatf_cf.CounterfactualExplainer(
    model=clf,
    dataset=iris_X,
    categorical_indices=[],
    default_numerical_step_size=0.1)


def describe_data_point(data_point_index):
    """Prints out a data point with the specified given index."""
    dp_to_explain = iris_X[data_point_index, :]
    dp_to_explain_class_index = int(iris_y[data_point_index])
    dp_to_explain_class = iris_class_names[dp_to_explain_class_index]

    feature_description_template = '    * {} (feature index {}): {:.1f}'
    features_description = []
    for i, name in enumerate(iris_feature_names):
        dsc = feature_description_template.format(name, i, dp_to_explain[i])
        features_description.append(dsc)
    features_description = ',\n'.join(features_description)

    data_point_description = (
        'Explaining data point (index {}) of class {} (class index {}) with '
        'features:\n{}.'.format(data_point_index, dp_to_explain_class,
                                dp_to_explain_class_index,
                                features_description))

    print(data_point_description)


###############################################################################
# Explain one of the data points.

# Select a data point to be explained
dp_1_index = 49
dp_1_X = iris_X[dp_1_index, :]
dp_1_y = iris_y[dp_1_index]
describe_data_point(dp_1_index)

# Get a Counterfactual Explanation tuple for this data point
dp_1_cf_tuple = cf_explainer.explain_instance(dp_1_X)
dp_1_cfs, dp_1_cfs_distances, dp_1_cfs_predictions = dp_1_cf_tuple
dp_1_cfs_predictions_names = np.array(
    [iris_class_names[i] for i in dp_1_cfs_predictions])

print('\nCounterfactuals for the data point:')
pprint(dp_1_cfs)
print('\nDistances between the counterfactuals and the data point:')
pprint(dp_1_cfs_distances)
print('\nClasses (indices and class names) of the counterfactuals:')
pprint(dp_1_cfs_predictions)
pprint(dp_1_cfs_predictions_names)

# Textualise the counterfactuals
dp_1_cfs_text = fatf_cf.textualise_counterfactuals(
    dp_1_X,
    dp_1_cfs,
    instance_class=dp_1_y,
    counterfactuals_distances=dp_1_cfs_distances,
    counterfactuals_predictions=dp_1_cfs_predictions)
print(dp_1_cfs_text)

###############################################################################
# Explain another data point.

# Select a data point to be explained
dp_2_index = 99
dp_2_X = iris_X[dp_2_index, :]
dp_2_y = iris_y[dp_2_index]
describe_data_point(dp_2_index)

# Get a Counterfactual Explanation tuple for this data point
dp_2_cf_tuple = cf_explainer.explain_instance(dp_2_X)
dp_2_cfs, dp_2_cfs_distances, dp_2_cfs_predictions = dp_2_cf_tuple
dp_2_cfs_predictions_names = np.array(
    [iris_class_names[i] for i in dp_2_cfs_predictions])

print('\nCounterfactuals for the data point:')
pprint(dp_2_cfs)
print('\nDistances between the counterfactuals and the data point:')
pprint(dp_2_cfs_distances)
print('\nClasses (indices and class names) of the counterfactuals:')
pprint(dp_2_cfs_predictions)
pprint(dp_2_cfs_predictions_names)

# Textualise the counterfactuals
dp_2_cfs_text = fatf_cf.textualise_counterfactuals(
    dp_2_X,
    dp_2_cfs,
    instance_class=dp_2_y,
    counterfactuals_distances=dp_2_cfs_distances,
    counterfactuals_predictions=dp_2_cfs_predictions)
print(dp_2_cfs_text)
