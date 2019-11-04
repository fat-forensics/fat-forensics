.. title:: About the Package

.. _about_the_package:

About the Package
+++++++++++++++++

FAT Forensics is a Python package -- supporting **Python 3.5** and up -- that
implements the state of the art **Fairness**, **Accountability** and
**Transparency** algorithms for the three main components of any
*data modelling* pipeline:

* **Data** (raw data and features),
* predictive **Models** and
* model **Predictions**.

It is distributed under the **3-Clause BSD License**, therefore opening it up
for commercial applications. The intention of the package is to collate all of
these algorithms under one roof with a unified structure and Application
Programming Interface (API).

FAT Forensics is different from all the other packages in this space as it is
being developed with the best software engineering practices in mind. Many
other packages in this space are a research output, therefore unintentionally
skewing their development towards particular research goals, hence forfeiting
flexibility. On the other hand, FAT Forensics is tested_ and supported by
:ref:`how to guides <how_to_guide>`, :ref:`tutorials <tutorials>`,
:ref:`user guides <user_guide>` and
:ref:`examples <sphx_glr_sphinx_gallery_auto>` on top of the
:ref:`raw API documentation <api_ref>`. All of this components contribute to a
package that is robust and easy to use, contribute and maintain.

Use Cases
=========

We envisage two main use cases for the package, each supported by distinct
features implemented to support it:

* an interactive *research* mode and
* a deployment mode for monitoring FAT aspects of a predictive system.

Interactive (Research)
----------------------

In a research mode -- data in, visualisation out -- the tool can be loaded into
an interactive Python session, e.g., a Jupyter Notebook, to support prototyping
and exploratory analysis. This mode is showcased in all of the
:ref:`examples <sphx_glr_sphinx_gallery_auto>` included in this documentation.
(Please note that you can download these examples as Jupyter Notebooks or
launch them in Binder_ with the URLs included at the bottom of each example.)

This mode is intended for FAT researchers who, for example, can use it to
propose new fairness metrics, compare them with the existing ones or use them
to inspect a new system or a data set.

Deployment (Monitoring)
-----------------------

In a deployment mode -- data in, data out -- the tool can be used as a part of
a data processing pipeline to provide a (numerical) FAT analytics, hence
support any kind of automated reporting or dashboarding. One such use case can
be building a monitoring dashboard with `Plotly's Dash`_ using FAT Forensics as
a back-end.

This mode is intended for Machine Learning engineers who may use it to monitor
or evaluate a Machine Learning system in pre-production, i.e. development and
deployment (train-deploy-monitor life-cycle), production (e.g., certification)
and post-production (e.g., reporting).

Structure of the Package
========================

The algorithms implemented by the package can be categorised along two main
axes: the **type of algorithm** (Fairness, Accountability and Transparency) and
the **part of a data processing pipeline** to which it is applicable (data,
models and predictions). The following table summarises this landscape and
names the main techniques implemented by FAT Forensics for each intersection.

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
|             |                           |                          | * Tabular bLIMEy         |
+-------------+---------------------------+--------------------------+--------------------------+

Fairness
--------

The *Fairness* part of the package is concerned with the broad field of
algorithmic fairness. It contains algorithms that can measure, evaluate and
correct for fairness for:

* data, both raw data and features extracted from data -- see the documentation
  of the :mod:`fatf.fairness.data` module,
* predictive models -- see the documentation of the :mod:`fatf.fairness.models`
  module -- and
* predictions -- see the documentation of the :mod:`fatf.fairness.predictions`
  module.

To learn more about the *Fairness* aspects of the FAT Forensics package please
consult the following parts of the documentation:

* :ref:`fairness examples <fairness_examples>`,
* :ref:`fairness user guides <user_guide_fairness>`,
* :ref:`fairness how to guides <how_to_fairness>` and
* :mod:`fatf.fairness` module API documentation.

Accountability
--------------

The *Accountability* part of the package is concerned with **safety**,
**security**, **robustness** and **privacy** of predictive systems. It contains
algorithms that can measure, evaluate and correct for these aspects for:

* data, both raw data and features extracted from data -- see the documentation
  of the :mod:`fatf.accountability.data` module,
* predictive models -- see the documentation of the
  :mod:`fatf.accountability.models` module -- and
* predictions -- see the documentation of the
  ``fatf.accountability.predictions`` module.

To learn more about the *Accountability* aspects of the FAT Forensics package
please consult the following parts of the documentation:

* :ref:`accountability examples <accountability_examples>`,
* :ref:`accountability user guides <user_guide_accountability>`,
* :ref:`accountability how to guides <how_to_accountability>` and
* :mod:`fatf.accountability` module API documentation.

Transparency
------------

The *Transparency* part of the package is concerned with **explainability**,
**interpretability** and **intelligibility** of predictive systems. It contains
algorithms that can peer inside and foster understanding of the following
aspects of predictive systems:

* data, both raw data and features extracted from data -- see the documentation
  of the :mod:`fatf.transparency.data` module,
* predictive models -- see the documentation of the
  :mod:`fatf.transparency.models` module -- and
* predictions -- see the documentation of the
  :mod:`fatf.transparency.predictions` module.

To learn more about the *Transparency* aspects of the FAT Forensics package
please consult the following parts of the documentation:

* :ref:`transparency examples <transparency_examples>`,
* :ref:`transparency user guides <user_guide_transparency>`,
* :ref:`transparency how to guides <how_to_transparency>` and
* :mod:`fatf.transparency` module API documentation.

.. _tested: https://travis-ci.com/fat-forensics/fat-forensics
.. _Binder: https://mybinder.org/v2/gh/fat-forensics/fat-forensics-doc/master?filepath=notebooks
.. _`Plotly's Dash`: https://plot.ly/dash/
