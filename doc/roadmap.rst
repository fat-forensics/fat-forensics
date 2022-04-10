.. title:: Roadmap

.. _roadmap:

Roadmap
+++++++

The following list of milestones is to guide the core developers on the future
direction of the package development. The list is by no means exhaustive and
will be updated over time as the development progresses and new algorithms
are proposed by the research community.

The list is algorithm- and feature-oriented as the goal of the package is to
give the community access to a tool that has all the necessary functionality
for FAT research and deployment.

Milestone 1 ✔
=============

The first milestone is our first public release of the package -- version
*0.0.1*. The following functionality should be available.

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

Milestone 2
===========

This will be the first major update of the package. The focus will be placed on
the transparency module. Nevertheless, some additional fairness and
accountability functionality will be implemented as well.

+-------------+---------------------------+--------------------------+--------------------------+
|             | Fairness                  | Accountability           | Transparency             |
+-------------+---------------------------+--------------------------+--------------------------+
| Data/       |                           | * k-anonymity            |                          |
| Features    |                           | * l-diversity            |                          |
|             |                           | * t-closeness            |                          |
+-------------+---------------------------+--------------------------+--------------------------+
| Models      | * Additional fairness     | * Background check       | * PD/ICE enhancements    |
|             |   metrics (to be          |                          | * Scikit-learn model     |
|             |   decided)                |                          |   explainers             |
|             |                           |                          | * ANCHOR                 |
|             |                           |                          | * Forestspy              |
|             |                           |                          | * Tree interpreter       |
|             |                           |                          | * Feature importance     |
|             |                           |                          | * Model reliance         |
|             |                           |                          | * TREPAN                 |
|             |                           |                          | * Logical models         |
|             |                           |                          |   counterfactual         |
|             |                           |                          |   explainer              |
|             |                           |                          |   for and their          |
|             |                           |                          |   ensembles              |
+-------------+---------------------------+--------------------------+--------------------------+
| Predictions |                           |                          | * Scikit-learn           |
|             |                           |                          |   prediction             |
|             |                           |                          |   explainers             |
|             |                           |                          | * Generalised local      |
|             |                           |                          |   surrogates (bLIMEy)    |
|             |                           |                          | * bLIMEy LIME            |
|             |                           |                          |   implementation for     |
|             |                           |                          |   tabular, text and      |
|             |                           |                          |   image data             |
+-------------+---------------------------+--------------------------+--------------------------+

- Extra fairness metrics.

  * Implement additional group-based fairness metrics.
  * Implement threshold computation based on the selected group metric
    equality.
  * Implement Jupyter Notebook interactive plugins (widgets) to allow the
    community to play with the fairness concepts. (E.g., widgets similar to
    interactive figures in this `Google blog post`_.

- Merge the pull request with k-anonimity, l-diversity and t-closeness.

- Implement `Background Check`_.

- PD and ICE enhancements (pull request).

  * 2-D implementation.
  * Implementation for classification and regression.
  * Improved visualisations.

- Scikit-learn model explainers (cf. the reference implementation in the
  `eli5 package`_).

  * Decision trees.

    + Feature importance.
    + Decision tree structure (tree plot).

  * Rule lists and sets (these can share a common representation with the
    trees).

    + Rule list structure (rule list in a text form).

  * Linear models.

    + Feature importance (coefficients).

  * K-means.

    + Prototypes.

      - Similarities between examples in a cluster that are correctly assigned
        to this clusetr.

    + Criticisms.

      - Similarities between examples in a cluster that are incorrectly
        assigned to this clusetr.

- Implement ANCHOR_.

- Implement forestspy_.

- Implement *Tree Interpreter*.

  * "The global feature importance of random forests can be quantified by the
    total decrease in node impurity averaged over all trees of the ensemble
    ('mean decrease impurity')."
  * "We can use the difference between the mean value of data points in a
    parent node between that of a child node to approximate the contribution of
    this split..."
  * `Interpreting random forests`_ and
    `Random forest interpretation with scikit-learn`_ blog posts hold some
    useful information extracted from the "Interpreting random forests" paper
    by Ando Saabas.

- Implement a variety of feature importance metrics.

  * Random forest feature (variable) importance ("Random Forests", Leo Breiman,
    2001). (Similar to *permutation importance*.)
  * XGboost feature importance.

    + Feature weight -- the number of times a feature appears in a tree
      (ensemble).
    + Gain -- the average gain of splits that use the feature.
    + Coverage -- the average coverage (number of samples affected) of splits
      that use the feature.

  * `Skater feature importance`_.
  * Prediction variance -- mean absolute value of changes in predictions given
    perturbations in the data.
  * "Variable Importance Analysis: A Comprehensive Review". Reliability
    Engineering & System Safety 142 (2015): 399-432; Wei, Pengfei, Zhenzhou Lu,
    and Jingwen Song.
  * Scikit-learn and eli5 **permutation importance** (a.k.a.
    *Mean Decrease Accuracy (MDA)*).

    + `eli5 implementation`_.
    + (These may be sensitive to features being correlated -- a user guide note
      should suffice.)

- Implement *model reliance* (Fisher, 2018). ("All Models are Wrong but many
  are Useful: Variable Importance for Black-Box, Proprietary, or Misspecified
  Prediction Models, using Model Class Reliance", Aaron Fisher, Cynthia Rudin,
  Francesca Dominici.)

- Implement TREPAN (tree surrogate).

  * "Extracting Comprehensible Models From Trained Neural Networks", Mark W.
    Craven(1996). (`PhD thesis`_)
  * "Extracting Thee-Structured Representations of Trained Networks", Mark W.
    Craven and Jude W. Shavlik (NIPS, 96). (`NIPS paper`_)
  * "Study of Various Decision Tree Pruning Methods with their Empirical
    Comparison in WEKA", Nikita Patel and Saurabh Upadhyay (2012). (report_)
  * `TREPAN implementation`_ in Skater.

- Implement a counterfactual explainer for logical models and their ensembles.

- Scikit-learn prediction explainers.

  * Decision trees.

    + Root-to-leaf path (logical conditions).
    + Counterfactuals.

  * Rule lists and sets.

    + Logical conditions list (as text).

  * Neighbours.

    + Similarities and differences (on the feature vector) among the neighbours
      of the same and the opposite class.

  * K-means.

    + Prototypes.

      - Nearest centroid of the same class.

    + Criticisms.

      - Nearest centroid of the opposite class.

- bLIMEy implementation.
- Fresh LIME implementation.

  * Write tutorials similar to `LIME tutorials`_, in particular
    `this tutorial`_.
  * Have a look at what eli5 does: "eli5.lime provides dataset generation
    utilities for text data (remove random words) and for arbitrary data
    (sampling using Kernel Density Estimation) ... for explaining predictions
    of probabilistic classifiers eli5 uses another classifier by default,
    trained using cross-entropy loss, while canonical library fits regression
    model on probability output."

.. _`Google blog post`: https://research.google.com/bigpicture/attacking-discrimination-in-ml/
.. _`Interpreting random forests`: https://blog.datadive.net/interpreting-random-forests/
.. _`Random forest interpretation with scikit-learn`: https://blog.datadive.net/random-forest-interpretation-with-scikit-learn/
.. _`Background Check`: https://github.com/perellonieto/background_check/blob/master/jupyter/background_check.ipynb
.. _`eli5 package`: https://eli5.readthedocs.io/en/latest/libraries/sklearn.html
.. _ANCHOR: https://github.com/marcotcr/anchor
.. _forestspy: https://github.com/jvns/forestspy
.. _`Skater feature importance`: https://oracle.github.io/Skater/reference/interpretation.html#feature-importance
.. _`eli5 implementation`: https://eli5.readthedocs.io/en/latest/autodocs/sklearn.html#module-eli5.sklearn.permutation_importance
.. _`PhD thesis`: https://ftp.cs.wisc.edu/machine-learning/shavlik-group/craven.thesis.pdf
.. _`NIPS paper`: https://papers.nips.cc/paper/1152-extracting-tree-structured-representations-of-trained-networks.pdf
.. _report: https://pdfs.semanticscholar.org/025b/8c109c38dc115024e97eb0ede5ea873fffdb.pdf
.. _`TREPAN implementation`: https://oracle.github.io/Skater/reference/interpretation.html#tree-surrogates-using-decision-trees
.. _`LIME tutorials`: https://github.com/marcotcr/lime/tree/gh-pages/tutorials
.. _`this tutorial`: https://marcotcr.github.io/lime/tutorials/Lime%20-%20multiclass.html

Milestone 3
===========

The third milestone will integrate the tool with important machine learning and
fairness packages.

+-------------+---------------------------+--------------------------+--------------------------+
|             | Fairness                  | Accountability           | Transparency             |
+-------------+---------------------------+--------------------------+--------------------------+
| Data/       |                           |                          |                          |
| Features    |                           |                          |                          |
+-------------+---------------------------+--------------------------+--------------------------+
| Models      | * Fairness360 integration | * Distribution shift     | * SHAP package           |
|             |                           |   detection              |   integration (Shapley   |
|             |                           | * Calibration            |   sampling values &      |
|             |                           |                          |   Shapley regression     |
|             |                           |                          |   values)                |
|             |                           |                          | * Xgboost package        |
|             |                           |                          |   interpreter            |
|             |                           |                          | * LightGBM package       |
|             |                           |                          |   interpreter            |
|             |                           |                          | * Lightning package      |
|             |                           |                          |   interpreter            |
|             |                           |                          | * Sklearn-crfsuite       |
|             |                           |                          |   package interpreter    |
|             |                           |                          | * eli5 package           |
|             |                           |                          |   integration            |
|             |                           |                          | * Bayesian Rule Lists    |
|             |                           |                          |   (BRL)                  |
|             |                           |                          | * PD/ICE speed           |
|             |                           |                          |   improvements           |
|             |                           |                          | * Interactive (JS)       |
|             |                           |                          |   Jupyter Notebook plots |
+-------------+---------------------------+--------------------------+--------------------------+
| Predictions |                           |                          | * SHAP package           |
|             |                           |                          |   integration            |
|             |                           |                          | * Xgboost package        |
|             |                           |                          |   interpreter            |
+-------------+---------------------------+--------------------------+--------------------------+

- Integration or reimplementation of fairness360_ package (depending on the
- code quality).
- Implement *distribution shift* metrics.
- Implement *calibration* techniques.
- Integration with the SHAP_ package.
- Explainers for models implemented in the Xgboost_ package.
- Explainers for models implemented in the LightGBM_ package.
- Explainers for models implemented in the Lightning_ package.
- Explainers for models implemented in the sklearn-crfsuite_ package.
- eli5_ integration. ("Text processing utilities from scikit-learn and can
  highlight text data accordingly. Pipeline and FeatureUnion are supported. It
  also allows to debug scikit-learn pipelines which contain HashingVectorizer,
  by undoing hashing.")
- Implement Bayesian Rule Lists (BRL).

  * Bayesian Rule Lists (BRL).

    + `BRL reference implementation`_.
    + Example `BRL use case`_: "Interpretable classifiers using rules and
      Bayesian analysis: Building a better stroke prediction model", Letham
      et.al(2015).

  * Scalable Bayesian Rule Lists (SBRL).

    + "Scalable Bayesian Rule Lists", Yang et.al (2016). (`SBRL paper`_)
    + Bayesian Rule List Classifier (BRLC_) is a Python wrapper for the SBRL.

  * Big Data Bayesian Rule List Classifier (BigDataBRLC) is a BRLC to handle
    large data-sets.

    + Skater's `BigDataBRLC implementation`_.
    + `Dr. Tamas Madl's implementation`_.

- PD/ICE speed improvements -- parallelisation and a progress bar.
- iPython/Jupyter Notebook interactive (JS) plots to improve research
  applicability aspect of the package.

.. _fairness360: https://github.com/IBM/AIF360
.. _SHAP: https://github.com/slundberg/shap
.. _Xgboost: https://github.com/dmlc/xgboost
.. _LightGBM: https://github.com/microsoft/LightGBM
.. _Lightning: https://github.com/scikit-learn-contrib/lightning
.. _sklearn-crfsuite: https://github.com/TeamHG-Memex/sklearn-crfsuite
.. _eli5: https://eli5.readthedocs.io/en/latest/libraries/index.html
.. _`BRL reference implementation`: https://oracle.github.io/Skater/reference/interpretation.html#bayesian-rule-lists-brl
.. _`BRL use case`: https://arxiv.org/abs/1511.01644
.. _`SBRL paper`: https://arxiv.org/abs/1602.08610
.. _BRLC: https://github.com/Hongyuy/sbrl-python-wrapper/blob/master/sbrl/C_sbrl.py
.. _`BigDataBRLC implementation`: https://oracle.github.io/Skater/reference/interpretation.html#skater.core.global_interpretation.interpretable_models.bigdatabrlc.BigDataBRLC
.. _`Dr. Tamas Madl's implementation`: https://github.com/tmadl/sklearn-expertsys/blob/master/BigDataRuleListClassifier.py

Milestone 4
===========

This milestone is focused on implementing in the package a collection of tools
that will enable researchers and practitioners to use it with (deep) neural
networks (Deep Learning, autograd, optimisation).

+-------------+---------------------------+--------------------------+--------------------------+
|             | Fairness                  | Accountability           | Transparency             |
+-------------+---------------------------+--------------------------+--------------------------+
| Data/       |                           |                          |                          |
| Features    |                           |                          |                          |
+-------------+---------------------------+--------------------------+--------------------------+
| Models      | * what-if tool            |                          |  * Quantitative Input    |
|             |   integration             |                          |    influence (QII)       |
|             |                           |                          |  * Layer-wise Relevance  |
|             |                           |                          |    Propagation (e-LRP)   |
|             |                           |                          |  * Occlusion             |
|             |                           |                          |  * integrated gradient   |
|             |                           |                          |  * what-if tool          |
|             |                           |                          |    integration           |
+-------------+---------------------------+--------------------------+--------------------------+
| Predictions |                           |                          | * DeepLIFT (example      |
|             |                           |                          |   explanation)           |
|             |                           |                          | * DeepExplain            |
+-------------+---------------------------+--------------------------+--------------------------+

- Integration with the `what-if tool`_.
- Implement Quantitative Input influence (QII).
- Implement epsilon-Layer-wise Relevance Propagation (e-LRP).

  * "On Pixel-Wise Explanations for Non-Linear Classifier Decisions by
    Layer-Wise Relevance Propagation", Bach S, Binder A, Montavon G, Klauschen
    F, Muller K-R, Samek W (2015).
  * "Towards better understanding of gradient-based attribution methods for
    Deep Neural Networks", Ancona M, Ceolini E, Oztireli C, Gross M (ICLR,
    2018).

- Implement *occlusion*.

  * "Visualizing and understanding convolutional networks", Zeiler, M and
    Fergus, R (Springer, 2014).
  * `Occlusion implementation`_.

- Implement Integrated Gradient method.

  * "Axiomatic Attribution for Deep Networks", Sundararajan, M, Taly, A, Yan, Q
    (ICML, 2017).
  * `Integrated Gradient slides`_.

- Implement the DeepLIFT algorithm.
- Implement the DeepExplain algorithm.

  * "Towards better understanding of gradient-based attribution methods for
    Deep Neural Networks", Ancona M, Ceolini E, Oztireli C, Gross M
    (ICLR, 2018).
  * `DeepExplain implementation`_.

----

* Finalise full integration of Skater and SHAP (deep neural netowrks).

.. _`what-if tool`: https://pair-code.github.io/what-if-tool/
.. _`Occlusion implementation`: https://github.com/marcoancona/DeepExplain/blob/master/deepexplain/tensorflow/methods.py
.. _`Integrated Gradient slides`: https://theory.stanford.edu/~ataly/Talks/sri_attribution_talk_jun_2017.pdf
.. _`DeepExplain implementation`: https://github.com/marcoancona/DeepExplain/blob/master/deepexplain/tensorflow/methods.py
