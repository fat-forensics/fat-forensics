"""
====================
Using LIME Explainer
====================

This example illustrates how to use the LIME tabular explainer in its generic
form to explain a prediction.

This example shows how to use the general LIME implementation (wrapper) --
:class:`fatf.transparency.lime.Lime`. However, the two sub-classes: one for a
model (:class:`fatf.transparency.models.lime.Lime`) and one for a prediction
(:class:`fatf.transparency.predictions.lime.Lime`) can be used in an almost
identical fashion to explain a model and a prediction correspondingly.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

from pprint import pprint

import fatf.utils.data.datasets as fatf_datasets
import fatf.utils.models as fatf_models

import fatf.transparency.lime as fatf_lime

import fatf.vis.lime as fatf_vis_lime

print(__doc__)

# Load data
iris_data_dict = fatf_datasets.load_iris()
iris_X = iris_data_dict['data']
iris_y = iris_data_dict['target']
iris_feature_names = iris_data_dict['feature_names']
iris_class_names = iris_data_dict['target_names']

# Train a model
clf = fatf_models.KNN()
clf.fit(iris_X, iris_y)

# Create a LIME explainer
lime = fatf_lime.Lime(
    iris_X,
    model=clf,
    feature_names=iris_feature_names,
    class_names=iris_class_names)

# Choose an index of the instance to be explained
index_to_explain = 42

# Explain an instance
lime_explanation = lime.explain_instance(
    iris_X[index_to_explain, :], num_samples=500)

# Display the textual explanation
pprint(lime_explanation)

# Plot the explanation
fatf_vis_lime.plot_lime(lime_explanation)
