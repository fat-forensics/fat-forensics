.. title:: Changelog

.. _changelog:

Changelog
+++++++++

.. _changelog_0_1_2:

0.1.2 (04/09/2022)
==================

The following bugs are fixed in this release:

- Segmentation:

  * A :class:`~fatf.utils.data.segmentation.Segmentation` object holds
    incorrect segment count after manipulation (`#39 <issue_39_>`_).
  * :class:`~fatf.utils.data.segmentation.Slic` segmentation fails quietly by
    not starting the segment count at 1.
    (This issue appears to have been fixed in scikit-image 0.19.2 and higher.)

- Occlusion:

  * `occlude_segments_vectorised`
    (:class:`~fatf.utils.data.occlusion.Occlusion`) returns an occlusion
    of incorrect shape if the input array is 2D with just one row
    (`#40 <issue_40_>`_).

.. _issue_39: https://github.com/fat-forensics/fat-forensics/issues/39
.. _issue_40: https://github.com/fat-forensics/fat-forensics/issues/40

.. _changelog_0_1_1:

0.1.1 (10/04/2022)
==================

The following functionality is made available with this release:

+-------------+----------+----------------+---------------------+
|             | Fairness | Accountability | Transparency        |
+-------------+----------+----------------+---------------------+
| Data/       |          |                |                     |
| Features    |          |                |                     |
+-------------+----------+----------------+---------------------+
| Models      |          |                | * Submodular        |
|             |          |                |   Pick              |
+-------------+----------+----------------+---------------------+
| Predictions |          |                | * Image bLIMEy      |
|             |          |                |   (LIME-equivalent) |
+-------------+----------+----------------+---------------------+

This update focuses on surrogate *image* explainers for predictions of
crisp and probabilistic black-box classifiers.
In particular, it implements:

- Segmentation:

  * Segmentation abstract class -- :class:`~fatf.utils.data.segmentation.\
    Segmentation`.
  * Slic segmentation -- :class:`~fatf.utils.data.segmentation.Slic`.
  * QuickShift segmentation -- :class:`~fatf.utils.data.segmentation.\
    QuickShift`.

- Occlusion:

  * Generic image occlusion -- :class:`~fatf.utils.data.occlusion.Occlusion`.

- Sampling:

  * Binary random sampling -- :func:`~fatf.utils.data.instance_augmentation.\
    random_binary_sampler`.

- Incremental model processing:

  * Batch-processing and -transforming data for predicting it with a model
    -- :func:`~fatf.utils.models.processing.batch_data`.

- Surrogate image explainability:

  * bLIMEy-based LIME surrogate image explainer -- :class:`~fatf.transparency.\
    predictions.surrogate_image_explainers.ImageBlimeyLime`.

- Aggregation-based model explainability:

  * Submodular pick -- :func:`~fatf.transparency.models.submodular_pick.\
    submodular_pick`.

Additionally, this release moves away from Travis CI in favour of GitHub
Actions.

.. _changelog_0_1_0:

0.1.0 (18/05/2020)
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
|             |          |                |   for regression |
+-------------+----------+----------------+------------------+

This is an incremental update focused on surrogate explainers for black-box
regression:

* Surrogate explainers -- :class:`~fatf.transparency.predictions.\
  surrogate_explainers.SurrogateTabularExplainer`, :class:`~fatf.transparency.\
  predictions.surrogate_explainers.TabularBlimeyLime` and :class:`~fatf.\
  transparency.predictions.surrogate_explainers.TabularBlimeyTree` -- now
  support black-box regression.
* :class:`~fatf.transparency.predictions.surrogate_explainers.\
  TabularBlimeyLime` now uses the correct feature selection approach.
* The surrogate explanation plotting function --
  :func:`~fatf.vis.lime.plot_lime` -- has been cleaned.
* 2 new feature selection approaches have been implemented:
  :func:`~fatf.utils.data.feature_selection.sklearn.highest_weights` and
  :func:`~fatf.utils.data.feature_selection.sklearn.forward_selection`.
* LIME wrapper has been removed.
* Compatibility with `scikit-learn` newer than `0.21.x` has been added.

This release coincides with publication of a `paper <JOSS_paper_>`_
describing FAT Forensic in The Journal of Open Source Software (JOSS).

.. _JOSS_paper: https://joss.theoj.org/papers/10.21105/joss.01904

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
  instead of ``fatf.transparency.predictions.lime.Lime``).

Included how-to guides:

* :ref:`how_to_tabular_surrogates`.

Included code examples:

* :ref:`sphx_glr_sphinx_gallery_auto_transparency_xmpl_transparency_lime.py`
  (updated to use
  :class:`fatf.transparency.predictions.surrogate_explainers.TabularBlimeyLime`
  instead of ``fatf.transparency.predictions.lime.Lime``) and
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
``fatf.transparency.lime.Lime`` will be retired in release 0.0.3.

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
