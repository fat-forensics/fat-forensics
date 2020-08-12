"""
==================================================
Using Permutation Feature Importance Explainer
==================================================

This example illustrates how to use Permutation Feature
Importance (PFI) Explainer.

* :func:`fatf.transparency.models.feature_influence.\
permutation_feature_importance`
* :ref:`tutorials_model_explainability`.

"""
# Author: Torty Sivill <vs14980@bristol.ac.uk>
# License: new BSD

import numpy as np
import matplotlib.pyplot as plt

import fatf.utils.data.datasets as fatf_datasets
import fatf.utils.models as fatf_models
import fatf.transparency.models.feature_influence as fatf_fi

print(__doc__)

# Load data
iris_data_dict = fatf_datasets.load_iris()
iris_X = iris_data_dict['data']
iris_y = iris_data_dict['target'].astype(int)
iris_feature_names = iris_data_dict['feature_names']

# Train a model
clf = fatf_models.KNN()
clf.fit(iris_X, iris_y)

# Choose the number of times to compute PFI
chosen_iterations = 5

# Choose a scoring metric from ``sklearn.metrics`` to be used to
chosen_metric = 'accuracy'

# Create a PFI explanation
pfi_scores = fatf_fi.permutation_feature_importance(
    clf,
    iris_X,
    iris_y,
    scoring_metric=chosen_metric,
    as_regressor=False,
    repeat_number=chosen_iterations)

print("PFI scores for each feature:")
for i, f_name in enumerate(iris_feature_names):
    print('{}: {}'.format(f_name, pfi_scores[:, i]))

# Get the mean and standard deviation over all
# iterations of each feature's PFI
mean_pfi_scores = np.mean(pfi_scores, axis=0)
std_pfi_scores = np.std(pfi_scores, axis=0)

print("Mean PFI scores for each feature:")
for i, f_name in enumerate(iris_feature_names):
    print('{}: {}'.format(f_name, mean_pfi_scores[i]))

print("Standard deviation of PFI scores for each feature:")
for i, f_name in enumerate(iris_feature_names):
    print('{}: {}'.format(f_name, std_pfi_scores[i]))

# Visualise the PFI scores with a boxplot where whiskers represent
# the range of PFI over different iterations
pfi_figure, pfi_axis = plt.subplots(1, 1)
pfi_axis.boxplot(pfi_scores)
pfi_axis.set_xticklabels(iris_feature_names)
pfi_axis.set_title("PFI for Iris Dataset")
pfi_axis.set_xlabel("Feature")
pfi_axis.set_ylabel("PFI")
plt.show()
