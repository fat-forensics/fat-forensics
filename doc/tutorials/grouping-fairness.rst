.. title:: Using Grouping to Evaluate Fairness of Data and Models

.. _tutorials_grouping_fairness:

Using Grouping to Evaluate Fairness of Data and Models -- Group-Based Fairness
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. topic:: Tutorial Contents

    In this tutorial, we show how to use the grouping functionality introduced
    in the previous tutorial together with the
    :func:`fatf.fairness.models.measures.disparate_impact_indexed` function to
    check for disparate treatment of a protected group with respect to the
    ground truth labelling (data fairness) and predictions provided by a model
    (model fairness).

Before we proceed let us load packages that we will need::

   >>> import numpy as np

For this tutorial we will use a small dataset distributed together with the
FAT Forensics package -- *health records*::

   >>> import fatf.utils.data.datasets as fatf_datasets
   >>> hr_data_dict = fatf_datasets.load_health_records()
   >>> hr_data = hr_data_dict['data']

This data set is stored as a `structured numpy array`_, therefore the columns
can be of different types (integers, strings, etc.) and they are accessible via
a name rather than an index.

The following features are available in this data set::

   >>> hr_data.dtype.names
   ('name', 'email', 'age', 'weight', 'gender', 'zipcode', 'diagnosis', 'dob')

Throughout this tutorial we will assume that the protected attribute is
``'gender'`` and we will inspect the distribution of the ground truth and
predictions of a model with respect to this feature in order to evaluate
fairness of this data set (its labelling in particular) and a model trained on
it.

Before we dive in let us briefly look at the ground truth vector::

   >>> hr_data_dict['target_names']
   array(['fail', 'success'], dtype='<U7')
   >>> hr_target = hr_data_dict['target']
   >>> np.unique(hr_target)
   array([0, 1])

Therefore, the target array has two possible values:

* ``0`` for a **fail**\ ed treatment and
* ``1`` for a **success**\ ful one.

Grouping the Data Set
=====================

In this section we will look into the distribution of the ground truth
(labeling) for sub-populations achieved by splitting the health records data
set based on the ``'gender'`` feature. To this end, let us use the data
grouping skills that we have learnt in the
:ref:`previous tutorial <tutorials_grouping>`::

   >>> import fatf.utils.data.tools as fatf_data_tools
   >>> gender_grouping = fatf_data_tools.group_by_column(hr_data, 'gender')
   >>> gender_grouping[1]
   ["('female',)", "('male',)"]
   >>> gender_grouping[0][0]
   [0, 1, 2, 6, 9, 12, 13, 14, 16, 17, 18, 19]
   >>> gender_grouping[0][1]
   [3, 4, 5, 7, 8, 10, 11, 15, 20]

Therefore, the data points with indices
``[0, 1, 2, 6, 9, 12, 13, 14, 16, 17, 18, 19]`` are ``'female'``\ s and the
data points with indices ``[3, 4, 5, 7, 8, 10, 11, 15, 20]`` are ``'male'``\ s.

.. _tutorials_grouping_fairness_labelling_disparity:

Labelling Disparity
===================

Now that we have ``'gender'``\ -based grouping we can investigate how target
labels are distributed for different genders. To this end, we will use numpy's
:func:`numpy.unique` with the ``return_counts`` parameter set to ``True``::

   >>> target_female = hr_target[gender_grouping[0][0]]
   >>> target_female_counts = np.unique(target_female, return_counts=True)
   >>> target_female_counts
   (array([0, 1]), array([5, 7]))

Therefore, for ``'female'``\ s there are 5 whose treatment has *failed* and 7
whose treatment was successful. For ``'male'``\ s the distribution of can be
computed in the same way::

   >>> target_male = hr_target[gender_grouping[0][1]]
   >>> target_male_counts = np.unique(target_male, return_counts=True)
   >>> target_male_counts
   (array([0, 1]), array([5, 4]))

Therefore, for ``'male'``\ s there are 5 whose treatment has failed and 4 whose
treatment was successful.

These look quite similar for both genders, which means that failed and
successful treatments distribution are comparable, therefore none of the gender
groups is underrepresented. We hope for the sub-populations to be similarly
distributed as an underrepresented group may cause a model to be under-fitted
in this region therefore underperform for this gender sub-population. This,
in turn, may lead to this sub-population being treated unfairly.

For completeness, let us compare these ratios numerically. The ratios for
females are::

   >>> female_fail_ratio = (
   ...     target_female_counts[1][0] / target_female_counts[1].sum())
   >>> female_fail_ratio
   0.4166666666666667
   >>> female_success_ratio = (
   ...     target_female_counts[1][1] / target_female_counts[1].sum())
   >>> female_success_ratio
   0.5833333333333334

And, the ratios for males are::

   >>> male_fail_ratio = (
   ...     target_male_counts[1][0] / target_male_counts[1].sum())
   >>> male_fail_ratio
   0.5555555555555556
   >>> male_success_ratio = (
   ...     target_male_counts[1][1] / target_male_counts[1].sum())
   >>> male_success_ratio
   0.4444444444444444

Therefore, the biggest ratio differences are::

   >>> abs(female_success_ratio - male_success_ratio)
   0.13888888888888895
   >>> abs(female_fail_ratio - male_fail_ratio)
   0.1388888888888889

which are acceptable if we assume a threshold of ``0.2``.

.. _tutorials_grouping_fairness_model_disparity:

Predictive Disparity of a Model
===============================

Now, let us inspect group-based fairness of a predictive model. To this end, we
first need to train a model::

   >>> import fatf.utils.models as fatf_models
   >>> clf = fatf_models.KNN()
   >>> clf.fit(hr_data, hr_target)

Next, we get the predictions for the training set::

   >>> hr_predictions = clf.predict(hr_data)

With that, we can see what is the training set accuracy of our model::

   >>> import fatf.utils.metrics.tools as fatf_metric_tools
   >>> import fatf.utils.metrics.metrics as fatf_performance_metrics

   >>> hr_confusion_matrix = fatf_metric_tools.get_confusion_matrix(
   ...     hr_target, hr_predictions)
   >>> fatf_performance_metrics.accuracy(hr_confusion_matrix)
   0.7619047619047619

The accuracy of ``0.76`` is not too bad. Now let us see how are accuracies
for males and females. First, we need to get confusion matrices for these two
sub-populations::

   >>> gender_cm = fatf_metric_tools.confusion_matrix_per_subgroup_indexed(
   ...     gender_grouping[0],
   ...     hr_target,
   ...     hr_predictions,
   ...     labels=np.unique(hr_target).tolist())

.. note::

   Please note that had we not coputed the groupings beforehand, we could use
   the :func:`fatf.utils.metrics.tools.confusion_matrix_per_subgroup` funciton,
   which computes the sub-populations automatically based on the feature
   indicated by the appropriate parameter. Please see the
   :ref:`sphx_glr_sphinx_gallery_auto_fairness_xmpl_fairness_models_measure.py`
   code example to see how it is used in practice.

Let us see the accuracy for females, which is the first sub-group::

   >>> fatf_performance_metrics.accuracy(gender_cm[0])
   0.8333333333333334

And, now, the one for males::

   >>> fatf_performance_metrics.accuracy(gender_cm[1])
   0.6666666666666666

.. module::
   fatf.fairness.models.measures
   :noindex:

The model at hand, clearly, performs better for females than it does for males.
But with the default threshold (``0.2``) this should should not be enough to
be indicated as violating the :func:`equal_accuracy` disparate impact fairness
metric::

   >>> import fatf.fairness.models.measures as fatf_fairness_models

   >>> gender_equal_accuracy = fatf_fairness_models.equal_accuracy(gender_cm)
   >>> fatf_fairness_models.disparate_impact_check(gender_equal_accuracy)
   False

As expected, none of the sub-population pairs -- in this case just one pair:
females-males -- violates the :func:`equal_accuracy` disparate impact fairness
metric, therefore the model is **fair** with respect to this metric with the
default threshold of ``0.2``.

Now, let us see whether it is also fair with respect the other two metrics
available in the package -- *equal opportunity* and *demographic parity*::

   >>> gender_equal_opportunity = fatf_fairness_models.equal_opportunity(
   ...     gender_cm)
   >>> fatf_fairness_models.disparate_impact_check(gender_equal_opportunity)
   False

   >>> gender_demographic_parity = fatf_fairness_models.demographic_parity(
   ...     gender_cm)
   >>> fatf_fairness_models.disparate_impact_check(gender_demographic_parity)
   True

These results indicate that for a threshold of ``0.2`` the *equal opportunity*
metric indicates fair treatment of both genders, whereas the
*demographic parity* is not satisfied. This clearly indicates that the choice
of a fairness metric (and a threshold) matters greatly and should always be
justified. Therefore, as a guidance for choosing the appropriate metric, one
should understand what it means for a given data set and modelling problem
before committing to it.

----

The concept of group-based fairness is strongly related to identifying
*protected* sub-populations in a data set and ensuring that a predictive model
does not underperform for one of them when compared to all the other. By
grouping a data set, on the other hand, we can see whether each of these
sub-groups is well represented, hence enable our predictive model to fit all of
them equally well.

The :ref:`next tutorial <tutorials_grouping_robustness>` shows how to use
grouping to inspect *accountability* of data and predictive models. Next, we
will move on to transparency of models and their predictions.

Relevant FAT Forensics Examples
===============================

The following examples provide more structured and code-focused use-cases of
the group-based fairness metrics and the data set fairness approaches:

* :ref:`sphx_glr_sphinx_gallery_auto_fairness_xmpl_fairness_models_measure.py`,
* :ref:`sphx_glr_sphinx_gallery_auto_fairness_xmpl_fairness_data_measure.py`.

.. _`structured numpy array`: https://docs.scipy.org/doc/numpy/user/basics.rec.html
