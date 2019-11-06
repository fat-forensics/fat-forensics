.. title:: Changelog

.. _changelog:

Changelog
+++++++++

.. _changelog_0_0_1:

0.0.1 (01/08/2019)
==================

This is the initial releases of the package. The following functionality is
made available with this release:

+-------------+---------------------------+--------------------------+--------------------------+
|             | Fairness                  | Accountability           | Transparency             |
+-------------+---------------------------+--------------------------+--------------------------+
| Data/       | * Systemic Bias           | * Sampling bias          | * Data description       |
| Features    |   (disparate treatment    | * Data Density Checker   |                          |
|             |   labelling)              |                          |                          |
|             | * Sample size disparity   |                          |                          |
|             |   (e.g., class imbalance) |                          |                          |
+-------------+---------------------------+--------------------------+--------------------------+
| Models      | * Group-based fairness    | * Systematic performance | * Partial dependence     |
|             |   (disparate impact)      |   bias                   | * Individual conditional |
|             |                           |                          |   expectation            |
+-------------+---------------------------+--------------------------+--------------------------+
| Predictions | * Counterfactual fairness |                          | * Counterfactuals        |
|             |   (disparate treatment)   |                          | * Tabular LIME (wrapper) |
+-------------+---------------------------+--------------------------+--------------------------+

.. _changelog_0_0_2:

0.0.2 (04/11/2019)
==================

The following functionality is made available with this release:

+-------------+----------+----------------+------------------+
|             | Fairness | Accountability | Transparency     |
+-------------+----------+----------------+------------------+
| Data/       |          |                |                  |
| Features    |          |                |                  |
+-------------+----------+----------------+------------------+
| Models      |          |                |                  |
+-------------+----------+----------------+------------------+
| Predictions |          |                | * Tabular bLIMEy |
+-------------+----------+----------------+------------------+

Included tutorials:

* :ref:`tutorials_prediction_explainability` (updated to use
  :class:`fatf.transparency.predictions.surrogate_explainers.TabularBlimeyLime`
  instead of :class:`fatf.transparency.predictions.lime.Lime`).

Included how-to guides:

* :ref:`how_to_tabular_surrogates`.

Included code examples:

* :ref:`sphx_glr_sphinx_gallery_auto_transparency_xmpl_transparency_lime.py`
  (updated to use
  :class:`fatf.transparency.predictions.surrogate_explainers.TabularBlimeyLime`
  instead of :class:`fatf.transparency.predictions.lime.Lime`) and
* :ref:`sphx_glr_sphinx_gallery_auto_transparency_xmpl_transparency_tree.py`.

bLIMEy
------

This release adds support for custom surrogate explainers of tabular data
called **bLIMEy**. The two pre-made classes are available as part of the
:mod:`fatf.transparency.predictions.surrogate_explainers` module:

* :class:`fatf.transparency.predictions.surrogate_explainers.\
  TabularBlimeyTree` and
* :class:`fatf.transparency.predictions.surrogate_explainers.\
  TabularBlimeyLime`.

Since the latter class implements LIME from components available in FAT
Forensics, the LIME wrapper available under
:class:`fatf.transparency.lime.Lime` will be retired in release 0.0.3.

To facilitate building custom tabular surrogate explainers a range of
functionality has been implemented including: data discretisation, data
transformation, data augmentation, data point augmentation, distance
kernelisation, scikit-learn model tools, feature selection and surrogate model
evaluation.

Other Functionality
-------------------

Seeding of the random number generators via the :func:`fatf.setup_random_seed`
function can now be done by passing a parameter to this function (in addition
to using the ``FATF_SEED`` system variable).
