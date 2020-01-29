---
title: |
  FAT Forensics: A Python Toolbox for Implementing and
  Deploying Fairness, Accountability and Transparency
  Algorithms in Predictive Systems
tags:
  - Fairness
  - Accountability
  - Transparency
  - Artificial Intelligence
  - Machine Learning
  - Software
  - Python Toolbox
authors:
  - name: Kacper Sokol
    orcid: 0000-0002-9869-5896
    affiliation: 1
  - name: Alexander Hepburn
    affiliation: 2
  - name: Rafael Poyiadzi
    affiliation: 2
  - name: Matthew Clifford
    affiliation: 2
  - name: Raul Santos-Rodriguez
    orcid: 0000-0001-9576-3905
    affiliation: 2
  - name: Peter Flach
    orcid: 0000-0001-6857-5810
    affiliation: 1
affiliations:
 - name: Department of Computer Science, University of Bristol
   index: 1
 - name: Department of Engineering Mathematics, University of Bristol
   index: 2
date: 01 September 2019
bibliography: paper.bib
---

# Background #

Predictive systems, in particular machine learning algorithms, can take
important, and sometimes legally binding, decisions about our everyday life. In
most cases, however, these systems and decisions are neither regulated nor
certified. Given the potential harm that these algorithms can cause, their
qualities such as **fairness**, **accountability** and **transparency** (FAT)
are of paramount importance. To ensure high-quality, fair, transparent and
reliable predictive systems, we developed an open source Python package called
*FAT Forensics*. It can inspect important fairness, accountability and
transparency aspects of predictive algorithms to automatically and objectively
report them back to engineers and users of such systems. Our toolbox can
evaluate all elements of a predictive pipeline: data (and their features),
models and predictions. Published under the BSD 3-Clause open source licence,
*FAT Forensics* is opened up for personal and commercial usage.

# Summary #

*FAT Forensics* is designed as an interoperable framework for *implementing*,
*testing* and *deploying* novel algorithms devised by the FAT research
community. It facilitates their evaluation and comparison against the state of
the art, thereby democratising access to these techniques. In addition to
supporting research in this space, the toolbox is capable of analysing all
components of a predictive pipeline -- data, models and predictions -- by
considering their fairness, accountability (robustness, security, safety and
privacy) and transparency (interpretability and explainability).

*FAT Forensics* collates all of these diverse tools and algorithms under a
common application programming interface. This is achieved with a modular
design that allows to share and reuse a collection of core algorithmic
components -- see Figure 1. This architecture makes the process of creating new
algorithms as easy as connecting the right blocks, therefore supporting a range
of diverse use cases.

![Modular architecture of FAT Forensics.](software_design.png){width=45%}

The format requirements for data sets and predictive models are kept to a
minimum, lowering any barriers for adoption of *FAT Forensics* in new and
already well-established projects. In this abstraction a data set is assumed to
be a two-dimensional NumPy array: either a classic or a structured array. The
latter is a welcome addition given that some of the features may be categorical
(string-based). A predictive model is assumed to be a plain Python object that
has `fit`, `predict` and, optionally, `predict_proba` methods. This flexibility
makes our package compatible with scikit-learn -- the most popular Python
machine learning toolbox -- without introducing additional dependencies.
Moreover, this approach makes *FAT Forensics* compatible with other packages
for predictive modelling since their predictive functions can be easily wrapped
inside a Python object with all the required methods.

Our package improves over existing solutions as it collates algorithms across
the FAT domains, taking advantage of their shared functional building blocks.
The common interface layer of the toolbox supports several
*modes of operation*. The **research mode** (data in -- visualisation out),
where the tool can be loaded into an interactive Python session, e.g., a
Jupyter Notebook, supports prototyping and exploratory analysis. This mode is
intended for FAT researchers who may use it to propose new fairness metrics,
compare them with the existing ones or use them to inspect a new system or a
data set. The **deployment mode** (data in -- data out) can be used as a part
of a data processing pipeline to provide a (numerical) FAT analytics,
supporting automated reporting and dashboarding. This mode is intended for
machine learning engineers and data scientists who may use it to monitor or
evaluate a predictive system during its development and deployment.

To encourage long-term maintainability, sustainability and extensibility,
*FAT Forensics* has been developed employing software engineering best practice
such as unit testing, continuous integration, well-defined package structure
and consistent code formatting. Furthermore, our toolbox is supported by a
thorough and beginner-friendly documentation that is based on four main
pillars, which together build up the user's confidence in using the package:

* narrative-driven **tutorials** designated for new users, which provide a
  step-by-step guidance through practical use cases of all the main aspects of
  the package;
* **how-to guides** created for relatively new users of the package, which
  showcase the flexibility of the toolbox and explain how to use it to solve
  user-specific FAT challenges; for example, how to build your own local
  surrogate explainer by pairing a data augmenter and an inherently transparent
  local model;
* **API documentation** describing functional aspects of the algorithms
  implemented in the package and designated for a technical audience as a
  reference material; it is complemented by task-focused *code examples* that
  put the functions, objects and methods in context; and
* a **user guide** discussing theoretical aspects of the algorithms implemented
  in the package such as their restrictions, caveats, computational time and
  memory complexity, among others.

We hope that this effort will encourage the FAT community to contribute their
algorithms to *FAT Forensics*. We offer it as an attractive alternative to
releasing yet more standalone packages, keeping the toolbox at the frontiers of
algorithmic fairness, accountability and transparency research. For a more
detailed description of *FAT Forensics*, we point the reader to its
documentation^[https://fat-forensics.org] and the paper [@sokol2019fatf]
describing its design, scope and usage examples.

# Related Work #

A recent attempt to create a common framework for FAT algorithms is the
*What-If* tool^[https://pair-code.github.io/what-if-tool], which implements
various fairness and explainability approaches. A number of Python packages
collating multiple state-of-the-art algorithms for either fairness,
accountability or transparency also exist. Available algorithmic *transparency*
packages include:

* Skater^[https://github.com/oracle/Skater] [@skater],
* ELI5^[https://github.com/TeamHG-Memex/eli5],
* Microsoft's Interpret^[https://github.com/interpretml/interpret]
  [@nori2019interpretml], and
* IBM's AI Explainability 360^[https://github.com/IBM/AIX360].

Packages implementing individual algorithms are also popular.
For example, LIME^[https://github.com/marcotcr/lime] for Local Interpretable
Model-agnostic Explanations [@ribeiro2016why] and
PyCEbox^[https://github.com/AustinRochford/PyCEbox] for Partial Dependence
[@friedman2001greedy] and Individual Conditional Expectation
[@goldstein2015peeking] plots.

Algorithmic *fairness* packages are also ubiquitous, for example: Microsoft's
fairlearn^[https://github.com/fairlearn/fairlearn] [@agarwal2018reductions] and
IBM's AI Fairness 360^[https://github.com/IBM/AIF360] [@bellamy2018ai].
However, *accountability* is relatively underexplored. The most prominent
software in this space deals with robustness of predictive systems against
adversarial attacks, for example:

* FoolBox^[https://github.com/bethgelab/foolbox],
* CleverHans^[https://github.com/tensorflow/cleverhans] and
* IBM's Adversarial Robustness 360
  Toolbox^[https://github.com/IBM/adversarial-robustness-toolbox].

*FAT Forensics* aims to bring together all of this functionality from across
fairness, accountability and transparency domains with its modular
implementation. This design principle enables the toolbox to support two
modes of operation: research and deployment. Therefore, the package caters
to a diverse audience and supports a range of tasks such as implementing,
testing and deploying FAT solutions. Abstracting away from fixed data set
and predictive model formats adds to its versatility. The development of the
toolbox adheres to best practices for software engineering and the package is
supported by a rich documentation, both of which make it stand out amongst
its peers.

# Acknowledgements #

This work was financially supported by Thales, and is the result of a
collaborative research agreement between Thales and the University of Bristol.

# References #
