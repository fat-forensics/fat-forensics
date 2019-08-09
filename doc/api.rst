.. title:: API Reference

.. _api_ref:

API Reference (|version|)
+++++++++++++++++++++++++

This is the class and function reference of fat-forensics. Please refer to
the :ref:`full user guide <user_guide>` for further details, as the class and
function raw specifications may not be enough to give full guidelines on their
uses.

.. note::

   The package is designed to work with both **classic** and **structured**
   numpy arrays. The latter is introduced to help manage numpy arrays holding
   vanila categorical features. Please see the
   :ref:`sphx_glr_sphinx_gallery_auto_fairness_xmpl_fairness_data_measure.py`
   and
   :ref:`sphx_glr_sphinx_gallery_auto_fairness_xmpl_fairness_models_measure.py`
   examples to see how the package can be used with a structured numpy array.

.. automodule:: fatf
    :no-members:
    :no-inherited-members:

----

.. _fairness_ref:

:mod:`fatf.fairness`: Fairness
==============================

.. automodule:: fatf.fairness
    :no-members:
    :no-inherited-members:

:mod:`fatf.fairness.data`: Fairness for Data
--------------------------------------------

.. automodule:: fatf.fairness.data
    :no-members:
    :no-inherited-members:

.. module:: fatf.fairness.data.measures

.. currentmodule:: fatf.fairness.data

.. autosummary::
   :toctree: generated/
   :template: function.rst
   :nosignatures:

   measures.systemic_bias
   measures.systemic_bias_check

:mod:`fatf.fairness.models`: Fairness for Models
------------------------------------------------

.. automodule:: fatf.fairness.models
    :no-members:
    :no-inherited-members:

.. module:: fatf.fairness.models.measures

.. currentmodule:: fatf.fairness.models

.. autosummary::
   :toctree: generated/
   :template: function.rst
   :nosignatures:

   measures.disparate_impact
   measures.disparate_impact_indexed
   measures.disparate_impact_check
   measures.demographic_parity
   measures.equal_opportunity
   measures.equal_accuracy

:mod:`fatf.fairness.predictions`: Fairness for Predictions
----------------------------------------------------------

.. automodule:: fatf.fairness.predictions
    :no-members:
    :no-inherited-members:

.. module:: fatf.fairness.predictions.measures

.. currentmodule:: fatf.fairness.predictions

.. autosummary::
   :toctree: generated/
   :template: function.rst
   :nosignatures:

   measures.counterfactual_fairness
   measures.counterfactual_fairness_check

.. _accountability_ref:

:mod:`fatf.accountability`: Accountability
==========================================

.. automodule:: fatf.accountability
    :no-members:
    :no-inherited-members:

:mod:`fatf.accountability.data`: Accountability for Data
--------------------------------------------------------

.. automodule:: fatf.accountability.data
    :no-members:
    :no-inherited-members:

.. module:: fatf.accountability.data.measures

.. currentmodule:: fatf.accountability.data

.. autosummary::
   :toctree: generated/
   :template: function.rst
   :nosignatures:

   measures.sampling_bias
   measures.sampling_bias_indexed
   measures.sampling_bias_grid_check
   measures.sampling_bias_check

:mod:`fatf.accountability.models`: Accountability for Models
------------------------------------------------------------

.. automodule:: fatf.accountability.models
    :no-members:
    :no-inherited-members:

.. module:: fatf.accountability.models.measures

.. currentmodule:: fatf.accountability.models

.. autosummary::
   :toctree: generated/
   :template: function.rst
   :nosignatures:

   measures.systematic_performance_bias
   measures.systematic_performance_bias_grid

.. _transparency_ref:

:mod:`fatf.transparency`: Transparency
======================================

.. automodule:: fatf.transparency
    :no-members:
    :no-inherited-members:

:mod:`fatf.transparency.data`: Transparency for Data
----------------------------------------------------

.. automodule:: fatf.transparency.data
    :no-members:
    :no-inherited-members:

.. module:: fatf.transparency.data.describe_functions

.. currentmodule:: fatf.transparency.data

.. autosummary::
   :toctree: generated/
   :template: function.rst
   :nosignatures:

   describe_functions.describe_array
   describe_functions.describe_numerical_array
   describe_functions.describe_categorical_array

:mod:`fatf.transparency.models`: Transparency for Models
---------------------------------------------------------

.. automodule:: fatf.transparency.models
    :no-members:
    :no-inherited-members:

:mod:`fatf.transparency.models.feature_influence`: Feature Influence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: fatf.transparency.models.feature_influence
    :no-members:
    :no-inherited-members:

.. currentmodule:: fatf.transparency.models

.. autosummary::
   :toctree: generated/
   :template: function.rst
   :nosignatures:

   feature_influence.individual_conditional_expectation
   feature_influence.merge_ice_arrays
   feature_influence.partial_dependence_ice
   feature_influence.partial_dependence

:mod:`fatf.transparency.models.lime`: LIME for Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: fatf.transparency.models.lime
    :no-members:
    :no-inherited-members:

.. currentmodule:: fatf.transparency.models

.. autosummary::
   :toctree: generated/
   :template: class.rst
   :nosignatures:

   lime.Lime

:mod:`fatf.transparency.predictions`: Transparency for Predictions
------------------------------------------------------------------

.. automodule:: fatf.transparency.predictions
    :no-members:
    :no-inherited-members:

:mod:`fatf.transparency.predictions.counterfactuals`: Counterfactual Explainers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: fatf.transparency.predictions.counterfactuals
    :no-members:
    :no-inherited-members:

.. currentmodule:: fatf.transparency.predictions

.. autosummary::
   :toctree: generated/
   :template: class.rst
   :nosignatures:

   counterfactuals.CounterfactualExplainer

.. autosummary::
   :toctree: generated/
   :template: function.rst
   :nosignatures:

   counterfactuals.textualise_counterfactuals

:mod:`fatf.transparency.predictions.lime`: LIME for Predictions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: fatf.transparency.predictions.lime
    :no-members:
    :no-inherited-members:

.. currentmodule:: fatf.transparency.predictions

.. autosummary::
   :toctree: generated/
   :template: class.rst
   :nosignatures:

   lime.Lime

:mod:`fatf.transparency.lime`: LIME Wrapper
-------------------------------------------

.. automodule:: fatf.transparency.lime
    :no-members:
    :no-inherited-members:

.. currentmodule:: fatf.transparency

.. autosummary::
   :toctree: generated/
   :template: class.rst
   :nosignatures:

   lime.Lime

:mod:`fatf.transparency.sklearn`: Scikit-learn Explainers
---------------------------------------------------------

.. automodule:: fatf.transparency.sklearn
    :no-members:
    :no-inherited-members:

.. _vis_ref:

:mod:`fatf.vis`: Visualisations
===============================

.. automodule:: fatf.vis
    :no-members:
    :no-inherited-members:

.. module:: fatf.vis.feature_influence
.. module:: fatf.vis.lime

.. currentmodule:: fatf.vis

.. autosummary::
   :toctree: generated/
   :template: function.rst
   :nosignatures:

   feature_influence.plot_individual_conditional_expectation
   feature_influence.plot_partial_dependence
   lime.plot_lime

.. _exceptions_ref:

:mod:`fatf.exceptions`: Exceptions, Errors and Warnings
=======================================================

.. automodule:: fatf.exceptions
    :no-members:
    :no-inherited-members:

.. currentmodule:: fatf.exceptions

.. autosummary::
   :toctree: generated/
   :template: exception.rst
   :nosignatures:

   FATFException
   IncorrectShapeError
   IncompatibleModelError
   UnfittedModelError
   PrefittedModelError

:mod:`fatf.utils`: Utilities
============================

.. automodule:: fatf.utils
    :no-members:
    :no-inherited-members:

**Developer guide:** See the :ref:`developers_guide` page for further details.

Base classes
------------

.. currentmodule:: fatf.utils

.. autosummary::
   :toctree: generated/
   :template: class.rst
   :nosignatures:

   data.augmentation.Augmentation
   models.models.Model

:mod:`fatf.utils.array`: Array Utilities
----------------------------------------

.. automodule:: fatf.utils.array
    :no-members:
    :no-inherited-members:

.. module:: fatf.utils.array.tools
.. module:: fatf.utils.array.validation

.. currentmodule:: fatf.utils.array

.. autosummary::
   :toctree: generated/
   :template: function.rst
   :nosignatures:

   tools.indices_by_type
   tools.get_invalid_indices
   tools.are_indices_valid
   tools.generalise_dtype
   tools.structured_to_unstructured_row
   tools.structured_to_unstructured
   tools.as_unstructured
   validation.is_numerical_dtype
   validation.is_textual_dtype
   validation.is_base_dtype
   validation.is_flat_dtype
   validation.are_similar_dtypes
   validation.are_similar_dtype_arrays
   validation.is_numerical_array
   validation.is_textual_array
   validation.is_base_array
   validation.is_1d_array
   validation.is_2d_array
   validation.is_structured_row
   validation.is_1d_like
   validation.is_structured_array

:mod:`fatf.utils.data`: Data Utilities
--------------------------------------

.. automodule:: fatf.utils.data
    :no-members:
    :no-inherited-members:

:mod:`fatf.utils.data.datasets`: Data Sets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: fatf.utils.data.datasets
    :no-members:
    :no-inherited-members:

.. currentmodule:: fatf.utils.data

.. autosummary::
   :toctree: generated/
   :template: function.rst
   :nosignatures:

   datasets.load_data
   datasets.load_health_records
   datasets.load_iris

:mod:`fatf.utils.data.tools`: Data Set Tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: fatf.utils.data.tools
    :no-members:
    :no-inherited-members:

.. currentmodule:: fatf.utils.data

.. autosummary::
   :toctree: generated/
   :template: function.rst
   :nosignatures:

   tools.group_by_column
   tools.apply_to_column_grouping
   tools.validate_indices_per_bin
   tools.validate_binary_matrix

:mod:`fatf.utils.data.augmentation`: Data Set Augmentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: fatf.utils.data.augmentation
    :no-members:
    :no-inherited-members:

.. currentmodule:: fatf.utils.data

.. autosummary::
   :toctree: generated/
   :template: class.rst
   :nosignatures:

   augmentation.NormalSampling
   augmentation.Mixup

:mod:`fatf.utils.data.density`: Data Set Density Checking and Estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: fatf.utils.data.density
    :no-members:
    :no-inherited-members:

.. currentmodule:: fatf.utils.data

.. autosummary::
   :toctree: generated/
   :template: class.rst
   :nosignatures:

   density.DensityCheck

:mod:`fatf.utils.models`: Models Utilities
------------------------------------------

.. automodule:: fatf.utils.models
    :no-members:
    :no-inherited-members:

:mod:`fatf.utils.models.models`: Example Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: fatf.utils.models.models
    :no-members:
    :no-inherited-members:

.. currentmodule:: fatf.utils.models

.. autosummary::
   :toctree: generated/
   :template: class.rst
   :nosignatures:

   models.KNN

:mod:`fatf.utils.models.validation`: Model Validation Tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: fatf.utils.models.validation
    :no-members:
    :no-inherited-members:

.. currentmodule:: fatf.utils.models

.. autosummary::
   :toctree: generated/
   :template: function.rst
   :nosignatures:

   validation.check_model_functionality

:mod:`fatf.utils.distances`: Distances and Distance Utilities
-------------------------------------------------------------

.. automodule:: fatf.utils.distances
    :no-members:
    :no-inherited-members:

.. currentmodule:: fatf.utils.distances

.. autosummary::
   :toctree: generated/
   :template: function.rst
   :nosignatures:

   get_distance_matrix
   get_point_distance
   euclidean_distance
   euclidean_point_distance
   euclidean_array_distance
   hamming_distance_base
   hamming_distance
   hamming_point_distance
   hamming_array_distance
   binary_distance
   binary_point_distance
   binary_array_distance

:mod:`fatf.utils.metrics`: Performance Metrics and Utilities
------------------------------------------------------------

.. automodule:: fatf.utils.metrics
    :no-members:
    :no-inherited-members:

:mod:`fatf.utils.metrics.metrics`: Basic Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: fatf.utils.metrics.metrics
    :no-members:
    :no-inherited-members:

.. currentmodule:: fatf.utils.metrics

.. autosummary::
   :toctree: generated/
   :template: function.rst
   :nosignatures:

   metrics.multiclass_true_positive_rate
   metrics.multiclass_true_negative_rate
   metrics.multiclass_false_positive_rate
   metrics.multiclass_false_negative_rate
   metrics.true_positive_rate
   metrics.true_negative_rate
   metrics.false_positive_rate
   metrics.false_negative_rate
   metrics.multiclass_positive_predictive_value
   metrics.multiclass_negative_predictive_value
   metrics.positive_predictive_value
   metrics.negative_predictive_value
   metrics.accuracy
   metrics.multiclass_treatment
   metrics.treatment

:mod:`fatf.utils.metrics.subgroup_metrics`: Metrics for Sub-Populations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: fatf.utils.metrics.subgroup_metrics
    :no-members:
    :no-inherited-members:

.. currentmodule:: fatf.utils.metrics

.. autosummary::
   :toctree: generated/
   :template: function.rst
   :nosignatures:

   subgroup_metrics.apply_metric_function
   subgroup_metrics.apply_metric
   subgroup_metrics.performance_per_subgroup
   subgroup_metrics.performance_per_subgroup_indexed

:mod:`fatf.utils.metrics.tools`: Metric Tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: fatf.utils.metrics.tools
    :no-members:
    :no-inherited-members:

.. currentmodule:: fatf.utils.metrics

.. autosummary::
   :toctree: generated/
   :template: function.rst
   :nosignatures:

   tools.get_confusion_matrix
   tools.confusion_matrix_per_subgroup
   tools.confusion_matrix_per_subgroup_indexed
   tools.validate_confusion_matrix
   tools.validate_confusion_matrix_size

:mod:`fatf.utils.tools`: FAT-Forensics Tools
--------------------------------------------

.. automodule:: fatf.utils.tools
    :no-members:
    :no-inherited-members:

.. currentmodule:: fatf.utils.tools

.. autosummary::
   :toctree: generated/
   :template: function.rst
   :nosignatures:

   at_least_verion

:mod:`fatf.utils.testing`: Testing Utilities
--------------------------------------------

.. automodule:: fatf.utils.testing
    :no-members:
    :no-inherited-members:

Constants
~~~~~~~~~

.. module:: fatf.utils.testing.arrays
.. module:: fatf.utils.testing.warnings

.. currentmodule:: fatf.utils.testing

.. autosummary::
   :toctree: generated/
   :nosignatures:

   arrays.NUMERICAL_NP_ARRAY
   arrays.NOT_NUMERICAL_NP_ARRAY
   arrays.WIDE_NP_ARRAY
   arrays.NUMERICAL_STRUCTURED_ARRAY
   arrays.NOT_NUMERICAL_STRUCTURED_ARRAY
   arrays.WIDE_STRUCTURED_ARRAY
   arrays.BASE_NP_ARRAY
   arrays.NOT_BASE_NP_ARRAY
   arrays.BASE_STRUCTURED_ARRAY
   arrays.NOT_BASE_STRUCTURED_ARRAY
   warnings.DEFAULT_WARNINGS
   warnings.EMPTY_RE
   warnings.EMPTY_RE_I

Functions
~~~~~~~~~

.. module:: fatf.utils.testing.imports
.. module:: fatf.utils.testing.vis

.. currentmodule:: fatf.utils.testing

.. autosummary::
   :toctree: generated/
   :template: function.rst
   :nosignatures:

   imports.module_import_tester
   vis.get_plot_data
   vis.get_line_data
   vis.get_bar_data
   warnings.handle_warnings_filter_pattern
   warnings.set_default_warning_filters
   warnings.is_warning_class_displayed

:mod:`fatf`: FAT Forensics Initialisation Functions
===================================================

This API documentation section describes *set-up* functionality of the FAT
Forensics package.

.. currentmodule:: fatf

.. autosummary::
   :toctree: generated/
   :template: function.rst
   :nosignatures:

   setup_warning_filters
   setup_random_seed
