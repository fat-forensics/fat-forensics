.. title:: Installation Instructions

.. _installation_instructions:

Installation Instructions
+++++++++++++++++++++++++

Installing the Package
======================

The package releases are uploaded to PyPi_, therefore allowing you to install
it with ``pip``:

.. code-block:: bash

   $ pip install fat-forensics

If you want to install the latest (development) version of the package please
see the instructions provided in the
:ref:`Developer Guide <developers_guide>`. Among others, it explains how to
install the package from source.

Package Requirements
====================

Hard Dependencies
-----------------

The package is built upon NumPy_ and SciPy_ with the following minimal
requirements:

+------------+------------+
| Package    | Version    |
+============+============+
| NumPy_     | 1.10.0     |
+------------+------------+
| SciPy_     | 0.13.3     |
+------------+------------+

These requirements are listed in the requirements.txt_ file in our GitHub
repository.

.. _installation_instructions_soft_dependencies:

Soft Dependencies
-----------------

Some parts (modules) of FAT Forensics depend on other packages. These are not
installed by default. The following table shows soft dependencies of ``fatf``
modules:

+-----------------------------------------------------------+------------------+
| ``fatf`` module                                           | Required package |
+===========================================================+==================+
| :mod:`fatf.transparency.predictions.surrogate_explainers` |                  |
+-----------------------------------------------------------+                  |
| :mod:`fatf.transparency.sklearn`                          | |scikit-learn|_  |
+-----------------------------------------------------------+                  |
| :mod:`fatf.utils.data.feature_selection.sklearn`          |                  |
+-----------------------------------------------------------+------------------+
| :mod:`fatf.vis`                                           | |matplotlib|_    |
+-----------------------------------------------------------+------------------+

These dependencies can either be installed manually or alongside
``fat-forensics`` via ``pip``:

.. code-block:: bash

   $ pip install fat-forensics[xxx]

where **xxx** can be replaced with any of the following to pull appropriate
soft dependencies during the package installation:

all
  Installs all soft dependencies: ``scikit-learn`` and ``matplotlib``.

ml
  Only installs ``scikit-learn`` as a soft dependency.

vis
  Only installs ``matplotlib`` as a soft dependency.

dev
  This option installs all the development requirements. Please consult the
  :ref:`Developer Guide <developers_guide>` for more details.

The exact versions of these soft dependencies can be found in the
requirements-aux.txt_ file.

Supported Platforms
===================

At the moments we only test the package on **Linux-based** systems. From our
experience the package also works on *Mac OS* and *Windows*, however these
platforms are not officially supported.

----

Recommended Reading Order
=========================

We recommend going through the documentation in the following order:

* the :ref:`description and some background <about_the_package>`,
* the :ref:`tutorials <tutorials>`, which will walk you through the basic
  concepts and tasks that can be solved with FAT Forensics step-by-step,
* the :ref:`examples <sphx_glr_sphinx_gallery_auto>`, which show a minimal
  code example for using every major functionality implemented by the package,
* the :ref:`user guide <user_guide>`, which describe the algorithms implemented
  by the package on a more conceptual level with their pros, cons, suggested
  applications, known shortcomings, best practices and alternative
  implementations,
* the :ref:`how to guides <how_to_guide>`, which show how to solve a particular
  problem with the package and, finally,
* the :ref:`API reference <api_ref>`, which should serve as a point of
  reference for every function and object within the package.

Developers and contributors may be interesting in the following pages as well:

* the :ref:`developers_guide` and
* the package development :ref:`roadmap`.

.. _PyPi: https://pypi.org/
.. _NumPy: https://numpy.org/
.. _SciPy: https://www.scipy.org/
.. _requirements.txt: https://github.com/fat-forensics/fat-forensics/blob/master/requirements.txt
.. |scikit-learn| replace:: ``scikit-learn>=0.19.2``
.. _scikit-learn: https://scikit-learn.org/stable/
.. |matplotlib| replace:: ``matplotlib>=3.0.0``
.. _matplotlib: https://matplotlib.org/
.. _requirements-aux.txt: https://github.com/fat-forensics/fat-forensics/blob/master/requirements-aux.txt
