.. -*- mode: rst -*-

=============  ================================================================
Software       |Licence|_ |GitHubRelease|_ |PyPi|_ |Python35|_
Docs           |Homepage|_
CI             |Travis|_ |Codecov|_
Try it         |Binder|_
Contact        |MailingList|_ |Gitter|_
Cite           |BibTeX|_ |JOSS|_
=============  ================================================================

.. |Licence| image:: https://img.shields.io/github/license/fat-forensics/fat-forensics.svg
.. _Licence: https://github.com/fat-forensics/fat-forensics/blob/master/LICENCE

.. |GitHubRelease| image:: https://img.shields.io/github/release/fat-forensics/fat-forensics.svg
.. _GitHubRelease: https://github.com/fat-forensics/fat-forensics/releases

.. |PyPi| image:: https://img.shields.io/pypi/v/fat-forensics.svg
.. _PyPi: https://pypi.org/project/fat-forensics/

.. |Python35| image:: https://img.shields.io/badge/python-3.5-blue.svg
.. _Python35: https://badge.fury.io/py/fat-forensics

.. .. |ReadTheDocs| image:: https://readthedocs.org/projects/fat-forensics/badge/?version=latest&style=flat
.. .. _ReadTheDocs: https://fat-forensics.readthedocs.io/en/latest/

.. |Homepage| image:: https://img.shields.io/badge/homepage-read-green.svg
.. _Homepage: https://fat-forensics.org
.. What about wiki?

.. |Travis| image:: https://travis-ci.com/fat-forensics/fat-forensics.svg?branch=master
.. _Travis: https://travis-ci.com/fat-forensics/fat-forensics

.. .. |CircleCI| image:: https://circleci.com/gh/fat-forensics/fat-forensics/tree/master.svg?style=shield
.. .. _CircleCI: https://circleci.com/gh/fat-forensics/fat-forensics/tree/master

.. |Codecov| image:: https://codecov.io/gh/fat-forensics/fat-forensics/branch/master/graph/badge.svg
.. _Codecov: https://codecov.io/gh/fat-forensics/fat-forensics

.. https://codeclimate.com/

.. https://requires.io/

.. |Binder| image:: https://mybinder.org/badge_logo.svg
.. _Binder: https://mybinder.org/v2/gh/fat-forensics/fat-forensics-doc/master?filepath=notebooks

.. |Docker| image:: https://images.microbadger.com/badges/image/anthropocentricai/ai-python.svg
.. _Docker: https://hub.docker.com/r/anthropocentricai/ai-python

.. |MailingList| image:: https://img.shields.io/badge/mailing%20list-Google%20Groups-green.svg
.. _MailingList: https://groups.google.com/forum/#!forum/fat-forensics

.. |Gitter| image:: https://img.shields.io/gitter/room/fat-forensics/FAT-Forensics.svg
.. _Gitter: https://gitter.im/fat-forensics

.. |BibTeX| image:: https://img.shields.io/badge/cite-BibTeX-blue.svg
.. _BibTeX: https://fat-forensics.org/getting_started/cite.html

.. |JOSS| image:: https://joss.theoj.org/papers/070c8b6b705bb47d1432673a1eb03f0c/status.svg
.. _JOSS: https://joss.theoj.org/papers/070c8b6b705bb47d1432673a1eb03f0c

.. |DOI| image:: https://zenodo.org/badge/DOI/xx.xxxx/zenodo.xxxxxxx.svg
.. _DOI: https://doi.org/xx.xxxx/zenodo.xxxxxxx

============================================================================
FAT Forensics: Algorithmic Fairness, Accountability and Transparency Toolbox
============================================================================

FAT Forensics (``fatf``) is a Python toolbox for evaluating fairness,
accountability and transparency of predictive systems. It is built on top of
SciPy_ and NumPy_, and is distributed under the 3-Clause BSD license (new BSD).

FAT Forensics implements the state of the art *fairness*, *accountability* and
*transparency* (FAT) algorithms for the three main components of any data
modelling pipeline: *data* (raw data and features), predictive *models* and
model *predictions*. We envisage two main use cases for the package, each
supported by distinct features implemented to support it: an interactive
*research mode* aimed at researchers who may want to use it for an exploratory
analysis and a *deployment mode* aimed at practitioners who may want to use it
for monitoring FAT aspects of a predictive system.

Please visit the project's web site `https://fat-forensics.org`_ for more
details.

Installation
============

Dependencies
------------

FAT Forensics requires **Python 3.5** or higher and the following dependencies:

+------------+------------+
| Package    | Version    |
+============+============+
| NumPy_     | >=1.10.0   |
+------------+------------+
| SciPy_     | >=0.13.3   |
+------------+------------+

In addition, some of the modules require *optional* dependencies:

+--------------------------------------------------------+------------------+------------+
| ``fatf`` module                                        | Package          | Version    |
+========================================================+==================+============+
| ``fatf.transparency.predictions.surrogate_explainers`` |                  |            |
+--------------------------------------------------------+                  |            |
| ``fatf.transparency.sklearn``                          | `scikit-learn`_  | >=0.19.2   |
+--------------------------------------------------------+                  |            |
| ``fatf.utils.data.feature_selection.sklearn``          |                  |            |
+--------------------------------------------------------+------------------+------------+
| ``fatf.vis``                                           | matplotlib_      | >=3.0.0    |
+--------------------------------------------------------+------------------+------------+

User Installation
-----------------

The easies way to install FAT Forensics is via ``pip``::

   pip install fat-forensics

which will only installed the required dependencies. If you want to install the
package together with all the auxiliary dependencies please consider using the
``[all]`` option::

   pip install fat-forensics[all]

The documentation provides more detailed `installation instructions <inst_>`_.

Changelog
=========

See the changelog_ for a development history and project milestones.

Development
===========

We welcome new contributors of all experience levels. The
`Development Guide <dev_guide_>`_ has detailed information about contributing
code, documentation, tests and more. Some basic development instructions are
included below.

Important Links
---------------

* Project's web site and documentation: `https://fat-forensics.org`_.
* Official source code repository:
  `https://github.com/fat-forensics/fat-forensics`_.
* FAT Forensics releases: `https://pypi.org/project/fat-forensics`_.
* Issue tracker: `https://github.com/fat-forensics/fat-forensics/issues`_.

Source Code
-----------

You can check out the latest FAT Forensics source code via git with the
command::

   git clone https://github.com/fat-forensics/fat-forensics.git

Contributing
------------

To learn more about contributing to FAT Forensics, please see our
`Contributing Guide <contrib_guide_>`_.

Testing
-------

You can launch the test suite from the root directory of this repository with::

   make test-with-code-coverage

To run the tests you will need to have version 3.9.1 of ``pytest`` installed.
This package, together with other development dependencies, can be also
installed with::

   pip install -r requirements-dev.txt

or with::

   pip install fat-forensics[dev]

See the *Testing* section of the `Development Guide <dev_testing_>`_ page for
more information.

    Please note that the ``make test-with-code-coverage`` command will test the
    version of the package in the local ``fatf`` directory and not the one
    installed since the pytest command is preceded by ``PYTHONPATH=./``. If
    you want to test the installed version, consider using the command from the
    ``Makefile`` without the ``PYTHONPATH`` variable.

    To control the randomness during the tests the ``Makefile`` sets the random
    seed to ``42`` by preceding each test command with ``FATF_SEED=42``, which
    sets the environment variable responsible for that. More information about
    the setup of the *Testing Environment* is available on the
    `development <dev_testing_env_>`_ web page in the documentation.

Submitting a Pull Request
-------------------------

Before opening a Pull Request, please have a look at the
`Contributing <contrib_guide_>`_ page to make sure that your code complies with
our guidelines.

Help and Support
================

For help please have a look at our
`documentation web page <https://fat-forensics.org>`_, especially the
`Getting Started <getting_started_>`_ page.

Communication
-------------

You can reach out to us at:

* our gitter_ channel for code-related development discussion; and
* our `mailing list`_ for discussion about the project's future and the
  direction of the development.

More information about the communication can be found in our documentation
on the `main page <https://fat-forensics.org/index.html#communication>`_ and
on the
`develop page <https://fat-forensics.org/development.html#communication>`_.

Citation
--------

If you use FAT Forensics in a scientific publication, we would appreciate
citations! Information on how to cite use is available on the
`citation <https://fat-forensics.org/getting_started/cite.html>`_ web page in
our documentation.

Acknowledgements
================
This project is the result of a collaborative research agreement between Thales
and the University of Bristol with the initial funding provided by Thales.

.. _SciPy: https://www.scipy.org/
.. _NumPy: https://www.numpy.org/
.. _scikit-learn: https://scikit-learn.org/stable/
.. _matplotlib: https://matplotlib.org/
.. _`https://fat-forensics.org`: https://fat-forensics.org
.. _inst: https://fat-forensics.org/getting_started/install_deps_os.html#installation-instructions
.. _changelog: https://fat-forensics.org/getting_started/changelog.html
.. _dev_guide: https://fat-forensics.org/development.html
.. _`https://github.com/fat-forensics/fat-forensics`: https://github.com/fat-forensics/fat-forensics
.. _`https://pypi.org/project/fat-forensics`: https://pypi.org/project/fat-forensics
.. _`https://github.com/fat-forensics/fat-forensics/issues`: https://github.com/fat-forensics/fat-forensics/issues
.. _contrib_guide: https://fat-forensics.org/development.html#contributing-code
.. _dev_testing: https://fat-forensics.org/development.html#testing
.. _dev_testing_env: https://fat-forensics.org/development.html#testing-environment
.. _getting_started: https://fat-forensics.org/getting_started/index.html
.. _gitter: https://gitter.im/fat-forensics
.. _`mailing list`: https://groups.google.com/forum/#!forum/fat-forensics
