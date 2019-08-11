.. title:: Using Grouping to Evaluate Robustness of Data and Models

.. _tutorials_grouping_robustness:

Using Grouping to Evaluate Robustness of Data and Models
++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. topic:: Tutorial Contents

    In this tutorial, we show how data grouping can be used to evaluate
    bias -- from the accountability perspective -- of a data set
    (*sampling bias*) and a predictive model (*systematic performance bias*).
    The former can help us to determine whether defined sub-populations are
    well represented in a data set -- similar to the
    :ref:`data fairness <tutorials_grouping_fairness_labelling_disparity>`
    consideration in the
    :ref:`previous tutorial <tutorials_grouping_fairness>`. The latter, can
    help us with identifying sub-populations in a data set for which a
    predictive model under-performs -- similar to the
    :ref:`model fairness <tutorials_grouping_fairness_model_disparity>`
    discussion in the :ref:`previous tutorial <tutorials_grouping_fairness>`.

First, we need to load numpy::

   >>> import numpy as np

Now, let us load and prepare the Iris data set::

   >>> import fatf.utils.data.datasets as fatf_datasets

   >>> iris_data_dict = fatf_datasets.load_iris()
   >>> iris_data = iris_data_dict['data']
   >>> iris_target = iris_data_dict['target'].astype(int)

.. note::

   For more information about the Iris data set and its structure, please refer
   the the :ref:`tutorials_grouping` tutorial or the `data set description`_
   on the `UCI repository`_ website.

Grouping the Data Set
=====================

For the purpose of this tutorial we will group the data set based on the third
feature of the data set::

   >>> iris_feature_names = iris_data_dict['feature_names']

   >>> selected_feature_index = 2
   >>> iris_feature_names[selected_feature_index]
   'petal length (cm)'

Now, let us assume that for some, unknown, reason there are two important
split values on this feature: ``2.5`` and ``4.75``::

   >>> import fatf.utils.data.tools as fatf_data_tools

   >>> selected_feature_groups = [2.5, 4.75]
   >>> selected_feature_grouping = fatf_data_tools.group_by_column(
   ...     iris_data,
   ...     selected_feature_index,
   ...     groupings=selected_feature_groups)
   >>> selected_feature_grouping[1]
   ['x <= 2.5', '2.5 < x <= 4.75', '4.75 < x']

Sampling Bias
=============

Given these two important splits we can now inspect these groupings and see
whether we have a comparable number of data points in each of them::

   >>> len(selected_feature_grouping[0][0])
   50
   >>> len(selected_feature_grouping[0][1])
   45
   >>> len(selected_feature_grouping[0][2])
   55

The number of data points for all the sub-populations seems to be (roughly)
equally distributed. The only pair of sub-populations which may indicate a
*sampling bias* is the second and the third one:
``2.5 < petal length (cm) <= 4.75`` and ``4.75 < petal length (cm)``. For
completeness, let us the
:func:`fatf.accountability.data.measures.sampling_bias_grid_check` function::

   >>> import fatf.accountability.data.measures as fatf_accountability_data

   >>> counts_per_grouping = [len(i) for i in selected_feature_grouping[0]]
   >>> fatf_accountability_data.sampling_bias_grid_check(counts_per_grouping)
   array([[False, False, False],
          [False, False,  True],
          [False,  True, False]])

As expected, the only pair of sub-populations violating *sampling bias*
criterion with the default threshold of ``0.8`` are sub-populations with
indices 1 and 2 making them: ``2.5 < petal length (cm) <= 4.75`` and
``4.75 < petal length (cm)``

.. note::

   Please note that the same result can be achieved without doing the data
   grouping manually. To this end, you may use the
   :func:`fatf.accountability.data.measures.sampling_bias` function, wchich
   internaly groups the data based on the specified feature index. The
   :ref:`sphx_glr_sphinx_gallery_auto_accountability_xmpl_accountability_data_measure.py`
   code example shows how to use it.

Systematic Performance Bias
===========================

Before we can evaluate robustness of a model, we first need one trained on the
Iris data set::

   >>> import fatf.utils.models as fatf_models
   >>> clf = fatf_models.KNN()
   >>> clf.fit(iris_data, iris_target)

We also need predictions of this model on a data set that we will use to
evaluate its robustness; in this case we will use the training data::

   >>> iris_pred = clf.predict(iris_data)

Before we can compute any performance metric, let us get confusion matrices for
each sub-population::

   >>> import fatf.utils.metrics.tools as fatf_metrics_tools

   >>> grouping_cm = fatf_metrics_tools.confusion_matrix_per_subgroup_indexed(
   ...     selected_feature_grouping[0],
   ...     iris_target,
   ...     iris_pred,
   ...     labels=np.unique(iris_target).tolist())


.. note:: UserWarning

   The above function call will generate 2 warnings::

      UserWarning: Some of the given labels are not present in either of the input arrays: {1, 2}.
      UserWarning: Some of the given labels are not present in either of the input arrays: {0}.

   These are because for some of the sub-populations the ground truth (target)
   and the prediction vectors may only hold a single label, therefore the
   confusion matrix calculator is not aware of the rest and has to resort to
   using the labels specified in the ``labels`` parameter. Printing the unique
   target and prediction values of the first sub-population shows exactly this
   scenario happening::

      >>> np.unique(iris_target[selected_feature_grouping[0][0]])
      array([0])
      >>> np.unique(iris_pred[selected_feature_grouping[0][0]])
      array([0])

   This happens as the selected feature -- petal length (cm) -- is a very good
   predictor of the first class. For more details you may want to have a look
   at the
   :ref:`data transparency section <tutorials_grouping_data_transparency>` of
   the :ref:`grouping tutorial <tutorials_grouping>` where this feature is
   explained in relation to the ground truth using the data descrition
   funcitonality of this package.

With confusion matrices for every grouping we can generate any performance
metric. For the purposes of this tutorial let us look at *accuracy*::

   >>> import fatf.utils.metrics.metrics as fatf_metrics

   >>> group_0_acc = fatf_metrics.accuracy(grouping_cm[0])
   >>> group_0_acc
   1.0
   >>> group_1_acc = fatf_metrics.accuracy(grouping_cm[1])
   >>> group_1_acc
   0.9777777777777777
   >>> group_2_acc = fatf_metrics.accuracy(grouping_cm[2])
   >>> group_2_acc
   0.9090909090909091

The accuracy seems to be comparable across sub-populations. Clearly none of
the sub-populations defined on the petal length feature suffers from a
performance bias as measured by accuracy. For completeness, let us test
for the systematic performance bias with the
:func:`fatf.accountability.models.measures.systematic_performance_bias_grid`
function::

   >>> import fatf.accountability.models.measures as fatf_accountability_models

   >>> fatf_accountability_models.systematic_performance_bias_grid(
   ...     [group_0_acc, group_1_acc, group_2_acc])
   array([[False, False, False],
          [False, False, False],
          [False, False, False]])

As expected, there is no systematic performance bias for these sub-populations
given the predictive model at hand.

.. note::

   In this part of the tutorial we used the
   :func:`fatf.utils.metrics.tools.confusion_matrix_per_subgroup_indexed`
   function to get a confusion matrix for each of the sub-populations and used
   these to compute the corresponding accuracies. All of these steps are
   combined by the
   :func:`fatf.utils.metrics.subgroup_metrics.performance_per_subgroup`
   function, therefore making the task of evaluating systematic performance
   bias easier. An example of how to use this function can be found in
   :ref:`sphx_glr_sphinx_gallery_auto_accountability_xmpl_accountability_models_measure.py`
   code example.

----

In this tutorial we saw how to use data grouping to evaluate important
accountability aspects of data sets and predictive models. This tutorial
concludes the series of tutorials focused around data grouping. In the next one
we move on to predictive models (:ref:`tutorials_model_explainability`) and
predictions (:ref:`tutorials_prediction_explainability`) transparency. For data
sets transparency please refer to the **last section** of the
:ref:`tutorials_grouping` tutorial.

Relevant FAT Forensics Examples
===============================

The following examples provide more structured and code-focused use-cases of
a group-based data and models inspection to evaluate their accountability:

* :ref:`sphx_glr_sphinx_gallery_auto_accountability_xmpl_accountability_data_measure.py`,
* :ref:`sphx_glr_sphinx_gallery_auto_accountability_xmpl_accountability_models_measure.py`.

.. _`data set description`: https://archive.ics.uci.edu/ml/datasets/iris
.. _`UCI repository`: https://archive.ics.uci.edu/ml/index.php
