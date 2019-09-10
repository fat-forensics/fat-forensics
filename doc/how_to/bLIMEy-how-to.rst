.. testsetup:: *

   >>> def fig_loc(loc):
   ...     import os
   ...     return os.path.join('doc/how_to/img', loc)
   >>> import os
   >>> os.environ['FATF_SEED'] = '42'
   >>> import fatf
   >>> fatf.setup_random_seed()

.. title:: bLIMEy How-To Guide

.. _bLIMEy_how_to:

bLIMEy: build LIME yourself Guide
+++++++++++++++++++++++++++++++++

.. topic:: How-to Guide Contents

    This how-to guide will show you how to construct a local linear model on
    top of a black-box classifier to be used to generate interpretable
    explanations. This how-to guide will require the scikit-learn
    dependency as it uses ridge regression and decision tree regressor as
    local models.

Let us load the functions from scikit-learn and numpy that we will need::

   >>> import numpy as np
   >>> import sklearn.model_selection
   >>> import sklearn.preprocessing
   >>> import sklearn.tree
   >>> import sklearn.metrics

Let us load the IRIS dataset and scale it between [0, 1] that we will be using::

   >>> import fatf.utils.data.datasets as fatf_datasets
   >>> iris_data_dict = fatf_datasets.load_iris()
   >>> iris_data = iris_data_dict['data']
   >>> iris_target = iris_data_dict['target']
   >>> iris_feature_names = iris_data_dict['feature_names']
   >>> iris_target_names = iris_data_dict['target_names']

In order for regression to return valid feature importances when training the
local model, we must scale the data between `[0, 1]`::

   >>> scaler = sklearn.preprocessing.MinMaxScaler(
   ...     feature_range=(0, 1)).fit(iris_data)
   >>> iris_data = scaler.transform(iris_data)
   >>> (iris_data.min(), iris_data.max())
   (0.0, 1.0000001)

Now we need to split our data into training and test splits::

   >>> train_data, test_data, train_labels, test_labels = \
   ...     sklearn.model_selection.train_test_split(
   ...         iris_data, iris_target, test_size=0.2, random_state=42)

Next, we have to train the k-nearest-neighbour classifier we will use as our
global model::

   >>> import fatf.utils.models.models as fatf_models
   >>> global_model = fatf_models.KNN(k=3)
   >>> global_model.fit(train_data, train_labels)
   >>> predictions = global_model.predict(test_data)
   >>> sklearn.metrics.accuracy_score(test_labels, predictions)
   1.0

As you can see, the IRIS dataset is reasonably easy and as such the KNN
classifier achieves a high accuracy. In order to generate an explanation for
this model, we need to choose the type of local surrogate we want to use. In
this guide we will use a decision tree as our local surrogate. The base LIME
algorithm uses ridge regression here.

First, we need to pick a point we wish to generate an explanation for::

   >>> data_point = iris_data[0]
   >>> data_point
   array([0.22222221, 0.62499994, 0.0677966 , 0.04166667], dtype=float32)

To show roughly where the ``data_point`` lies in the dataset, lets plot the
first two dimensions of the data and highlight the ``data_point``::

   >>> import matplotlib.pyplot as plt
   >>> _ = plt.scatter(
   ...     iris_data[1:50, 0],
   ...     iris_data[1:50, 1],
   ...     label=iris_target_names[0])
   >>> _ = plt.scatter(
   ...     iris_data[50:100, 0],
   ...     iris_data[50:100, 1],
   ...     label=iris_target_names[1])
   >>> _ = plt.scatter(
   ...     iris_data[100:150, 0],
   ...     iris_data[100:150, 1],
   ...     label=iris_target_names[2])
   >>> _ = plt.scatter(
   ...     data_point[0],
   ...     data_point[1],
   ...     label='Explanation Point',
   ...     s=200, c='k')
   >>> _ = plt.legend()
   >>> _ = plt.xlim(0.0, 1.0)
   >>> _ = plt.ylim(0.0, 1.0)

.. testsetup:: *

   >>> plt.savefig(fig_loc('iris_plot_explanation.png'), dpi=100)

.. image:: /how_to/img/iris_plot_explanation.png
   :align: center
   :scale: 75

We use an augmentor in :mod:`fatf.utils.data.augmentation` to generate data
locally around the point. We will be using the :class:`fatf.utils.data.\
augmentation.Mixup` augmentor. For a more detailed explanation of each
augmentor please see :mod:fatf.utils.data.augmentation`::

   >>> import fatf.utils.data.augmentation as fatf_augmentation
   >>> augmentor = fatf_augmentation.Mixup(train_data, train_labels)
   >>> augmented_data = augmentor.sample(data_point, samples_number=50)

Lets plot the first two dimensions of the augmented data and compare it to
the original dataset::

   >>> _ = plt.figure()
   >>> _ = plt.scatter(
   ...      augmented_data[:, 0],
   ...      augmented_data[:, 1],
   ...      label='Augmented Data')
   >>> _ = plt.scatter(
   ...      data_point[0],
   ...      data_point[1],
   ...      label='Explanation Point',
   ...      s=200, c='k')
   >>> _ = plt.legend()
   >>> _ = plt.xlim(0.0, 1.0)
   >>> _ = plt.ylim(0.0, 1.0)

.. testsetup:: *

   >>> plt.savefig(fig_loc('iris_plot_augmented.png'), dpi=100)

.. image:: /how_to/img/iris_plot_augmented.png
   :align: center
   :scale: 75

In this guide, we will be explaining the prediction for ``data_point`` for the
`setosa` class which corresponds to class 0. In order to train a local
regression model, we need the predicted probabilities from the global model
that each sample belongs to class 0::

  >>> probabilities = global_model.predict_proba(augmented_data)[:, 0]

If we use a decision tree regressor, discretisation and feature selection will
performed automatically. As such we can just train the local model and
extract the most important features and their weights::

   >>> local_model = sklearn.tree.DecisionTreeRegressor(
   ...     max_depth=3, random_state=42)
   >>> _ = local_model.fit(augmented_data, probabilities)
   >>> tree_as_text = sklearn.tree.export.export_text(
   ...     local_model, feature_names=list(iris_data_dict['feature_names']))
   >>> print(tree_as_text)
   |--- petal length (cm) <= 0.31
   |   |--- sepal width (cm) <= 0.35
   |   |   |--- value: [0.00]
   |   |--- sepal width (cm) >  0.35
   |   |   |--- petal width (cm) <= 0.24
   |   |   |   |--- value: [1.00]
   |   |   |--- petal width (cm) >  0.24
   |   |   |   |--- value: [0.83]
   |--- petal length (cm) >  0.31
   |   |--- sepal length (cm) <= 0.27
   |   |   |--- sepal width (cm) <= 0.46
   |   |   |   |--- value: [0.00]
   |   |   |--- sepal width (cm) >  0.46
   |   |   |   |--- value: [0.67]
   |   |--- sepal length (cm) >  0.27
   |   |   |--- value: [0.00]
   <BLANKLINE>
