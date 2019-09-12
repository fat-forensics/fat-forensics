.. title:: Developers Guide

.. _developers_guide:

Developers Guide
++++++++++++++++

In this guide you will find all the information you need to contribute high
quality code and documentation to the package.

Communication
=============

We use a range of platforms for communication in the project:

* for issues with the source code or the documentation please open an issue on
  our `GitHub issue tracker`;
* the code-related development discussion should happen on our
  `gitter channel`_;
* the discussion about the project's future and the direction of the
  development happens on our `slack channel`_ and `mailing list`_.

.. _`GitHub issue tracker`: https://github.com/fat-forensics/fat-forensics/issues
.. _`gitter channel`: https://gitter.im/fat-forensics/fat-forensics
.. _`slack channel`: https://fat-forensics.slack.com
.. _`mailing list`: https://groups.google.com/forum/#!forum/fat-forensics

Contributing Code
=================

There is a number of steps that any new code or documentation patch has to go
through before being merged into the package. All of these are described below.

.. _developers_guide_dev_requirements:

Installing Development Requirements
-----------------------------------

The python tools and packages used for the development and testing of the code
need to be installed before proceeding. In order to run all of the tests all:
the *required* package dependencies, the *auxiliary* package dependencies and
the *development* package dependencies need to be installed.

Files with these requirements can be found in the root folder of the package
repository under the following names:

* requirements.txt_ -- the dependencies required by the package;
* requirements-aux.txt_ -- the soft dependencies of the package (cf.
  :ref:`installation_instructions_soft_dependencies`); and
* requirements-dev.txt_ -- the development dependencies of the package.

All of these can be installed directly from these files with

.. code-block:: bash

   $ pip install -r <dependencies file name>

or alongside the package with

.. code-block:: bash

   $ pip install -e fat-forensics[all,dev]

given that your current directory is the root directory of the package. (See
the section below for more information.)

.. _requirements.txt: https://github.com/fat-forensics/fat-forensics/blob/master/requirements.txt
.. _requirements-aux.txt: https://github.com/fat-forensics/fat-forensics/blob/master/requirements-aux.txt
.. _requirements-dev.txt: https://github.com/fat-forensics/fat-forensics/blob/master/requirements-dev.txt

Installing the Package
----------------------

When developing code for the package we advise to install it as an
**editable copy** directly from sources placed in the ``dev`` branch. To
achieve that you need to first clone our git repository and then install FAT
Forensics as an editable package alongside all the required dependencies. To
this end, please execute the four commands shown below.

.. code-block:: bash

   $ git clone https://github.com/fat-forensics/fat-forensics.git
   $ cd fat-forensics
   $ git checkout dev
   $ pip install -e '.[all,dev]'

.. note::

   Consider using pyenv_ and pyenv-virtualenv_ plugin to manage your Python
   versions and virtual environments. It makes the development so much easier.

If you want to install the latest *stable* version from sources instead, please
use the ``master`` branch instead.

.. _pyenv: https://github.com/pyenv/pyenv
.. _pyenv-virtualenv: https://github.com/pyenv/pyenv-virtualenv

Testing Environment
-------------------

We develop the package so that it is compatible with Python versions 3.5 and
up. The specific versions of the package dependencies are listed in the three
requirement files listed in the :ref:`developers_guide_dev_requirements`
section.

All of the test and build commands are held in the Makefile_, which is placed
in the root directory of the package. Please consult this file for the specific
commands that are executed when building and testing the package.

Our Continuous Integration (CI) is run on Travis_ with the configuration file
-- `.travis.yml`_ -- held in the root directory of the package.

.. note:: **Random number generator**

   Since some of the package's tests sample random numbers or are influenced by
   "randomness" of its dependencies, the package allows to fix the random seed
   of both Python's :mod:`random` and numpy's :mod:`numpy.random` random number
   generation modules. This can be achieved by setting the ``FATF_SEED`` system
   variable to a selected number. (For tests this is set to 42 in the
   Makefile_.) If set, the :func:`fatf.setup_random_seed` function, which is
   called upon importing the FAT Forensics package, will set both of the
   aforementioned random seeds to the desired value.

   If anywhere in the tests you wish to restore it to the desired value,
   calling the :func:`fatf.setup_random_seed` function should suffice.

.. _Makefile: https://github.com/fat-forensics/fat-forensics/blob/master/Makefile
.. _Travis: https://travis-ci.com/fat-forensics/fat-forensics
.. _`.travis.yml`: https://github.com/fat-forensics/fat-forensics/blob/master/.travis.yml

Code Formatting
---------------

When writing the code we try to follow the `Google Python Style Guide`_.
Code formatting adherence to this guideline can be checked with
*Yet Another Python Formater* (YAPF_) by executing:

.. code-block:: bash

   $ make linting-yapf

which will highlight what needs to be changed rather than reformat the code
automatically. The configuration of YAPF_ for the package can be found in the
`.style.yapf`_ file in the root directory of the package.

Code formatting is also checked with Pylint_ and Flake8_. These can be executed
with the following two commands:

.. code-block:: bash

   $ make linting-pylint
   $ make linting-flake8

The configuration of both these linters can be found in `.pylintrc`_ and
`.flake8`_ files respectively, both placed in the root directory of the
package.

To help the contributors adhere to the formatting style of the code,
documentation and configuration files we use EditorConfig_. By installing the
EditorConfig_ plugin for your code editor the style of the new content that you
author will automatically adhere to some of our coding style. You can find the
configuration file of the EditorConfig -- `.editorconfig`_ -- in the root
directory of the package.

.. _`Google Python Style Guide`: http://google.github.io/styleguide/pyguide.html
.. _YAPF: https://github.com/google/yapf
.. _`.style.yapf`: https://github.com/fat-forensics/fat-forensics/blob/master/.style.yapf
.. _Pylint: https://www.pylint.org/
.. _Flake8: http://flake8.pycqa.org/en/latest/
.. _`.pylintrc`: https://github.com/fat-forensics/fat-forensics/blob/master/.pylintrc
.. _`.flake8`: https://github.com/fat-forensics/fat-forensics/blob/master/.flake8
.. _EditorConfig: https://editorconfig.org
.. _`.editorconfig`: https://github.com/fat-forensics/fat-forensics/blob/master/.editorconfig

Type Hints
----------

We try to annotate the code in the package with type hints whenever possible.
The typing of the code is checked statically with mypy_. Our mypy configuration
file -- `.mypy.ini`_ -- is placed in the root directory of the package and the
type checking is performed by executing the following line of code:

.. code-block:: bash

   $ make check-types

.. _mypy: http://mypy-lang.org/
.. _`.mypy.ini`: https://github.com/fat-forensics/fat-forensics/blob/master/.mypy.ini

Testing
-------

We run tests on the package itself as well as on the code snippets spread
throughout the documentation. To this end, we use pytest_ configured with the
`pytest.ini`_ file kept in the root directory of the package.

To gather code coverage statistics we use pytest-cov_ plugin with its partial
configuration placed in the `.coveragerc`_ file kept in the root directory of
the package.

Code
~~~~

The unit tests for the package are held in directories named ``tests`` created
separately for each module. To test the code you can execute:

.. code-block:: bash

   $ make test

and to get the code coverage:

.. code-block:: bash

   $ make code-coverage

However, we recommend to execute both these steps at once to save time by
using:

.. code-block:: bash

   $ make test-with-code-coverage

.. note::

   The :mod:`fatf.utils.testing` module holds a range of functions that are
   useful for the unit tests. If you find yourself reusing a piece of code
   in multiple places in the unit tests, please consider making it a part of
   this module.

Documentation
~~~~~~~~~~~~~

There are three different tests run on the documentation. The first one checks
validity of links in the documentation and is run with:

.. code-block:: bash

   $ make doc-linkcheck

The second one checks which Python objects, methods and functions are not
documented (documentation coverage) and can be run with:

.. code-block:: bash

   $ make doc-coverage

Finally, the code snippets spread throughout the documentation are run to test
whether their output agrees with the one provided in the documentation. These
tests can be run with:

.. code-block:: bash

   $ make test-doc

.. note::

   Because of incompatibility of vanila (pytest) doctest and sphinx doctest
   we are using pure doctest syntax, i.e. no group annotations are possible.

.. _pytest: https://pytest.org/en/latest/
.. _`pytest.ini`: https://github.com/fat-forensics/fat-forensics/blob/master/pytest.ini
.. _pytest-cov: https://pytest-cov.readthedocs.io/en/latest/
.. _`.coveragerc`: https://github.com/fat-forensics/fat-forensics/blob/master/.coveragerc

Step by Step Guide
==================

To summarise, the following commands should be executed to fully test the
package (cf. `.travis.yml`_ for more details):

.. code-block:: bash

   $ make linting-pylint
   $ make linting-flake8
   $ make linting-yapf

   $ make check-types

   $ make test-with-code-coverage

   $ make doc-linkcheck
   $ make doc-coverage
   $ make test-doc

Contributing Documentation
==========================

To understand the documentation structure and the intention of each section in
the documentation please familiarise yourself with the :ref:`getting_started`
part of the documentation.

Code Documentation
------------------

For building the documentation we use Sphinx_ with a `custom theme`_. The API
is documented using numpydoc_ docstring syntax and structure. The code example
galleries are generated with the sphinx-gallery_ extension.

To build the documentation you can execute:

.. code-block:: bash

   $ make doc-html

In addition to this command being available in the main Makefile_ of the
project, there is a `documentation-specific Makefile`_ in the ``doc`` directory
that supports the following documentation build command:

.. code-block:: bash

   $ make html

Since some of the code snippets (in particular the ones placed in the
tutorials) produce plots and figures that are later included in the
documentation, these need to be executed first. To this end, the documentation
tests (``make test-doc``) has to be executed before building the documentation.

.. warning::

   Since the ``.rst`` files describing the API documentation are generated
   automatically with sphinx's ``autosummary`` extension and placed in the
   ``doc/generated`` directory some of the changes that are made to the API
   template or documentation may not trigger the automatic rebuilding of the
   generated files. In such cases the ``doc/generated`` directory has to be
   cleaned. This can be achieved with the `documentation-specific Makefile`_
   via the following command:

   .. code-block:: bash

      $ make doc-clean

User Guide
----------

In addition to the FAT Forensics
:ref:`package-oriented documentation <getting_started>` we also maintain a
:ref:`user_guide` that describes :ref:`Fairness <user_guide_fairness>`,
:ref:`Accountability <user_guide_accountability>` and
:ref:`Transparency <user_guide_transparency>` approaches on a more theoretical
level. Entries in the :ref:`user_guide` should try to follow a specific
fact-oriented pattern. When contributing please try to adhere to the style of
the entries that are already in the :ref:`user_guide` as much as possible.
At a minimum please provide the following fields in the method description
placed in the :ref:`user_guide`:

- Name.
- Literature reference (BibTeX).
- List of implementations (both standalone implementations and implementations
  in packages and libraries).

  * Programming language.
  * Implementation URL (possibly GitHub).
  * Licence.
  * **F**\ airness, **A**\ ccountability, **T**\ ransparency or **\*** (other
    and related) field.

- Is is a *Metric* (measure) or an *Application* (mitigation) technique.
- Is it *Model Dependent* (what are the applicable models) or *Model Agnostic*.
- Is it *Post-Hoc* or *Ante-Hoc*.

.. _Sphinx: http://www.sphinx-doc.org/en/master/
.. _`custom theme`: https://github.com/fat-forensics/fat-forensics/tree/master/doc/themes/fat-forensics
.. _numpydoc: https://numpydoc.readthedocs.io/en/latest/
.. _sphinx-gallery: https://sphinx-gallery.github.io/
.. _`documentation-specific Makefile`: https://github.com/fat-forensics/fat-forensics/blob/master/doc/Makefile

Pull Requests and Issues
========================

When making a pull request on GitHub please use the provided
`pull request template`_ and make sure that you comply with all the
requirements listed therein. Furthermore, please have a browse through other
`pull requests`_ and issues_ to locate all the problems/solutions similar to
yours.

Similarly, we have `issue templates`_. Please use them (whenever possible)
while opening a new issue.

.. _`pull request template`: https://github.com/fat-forensics/fat-forensics/blob/master/.github/PULL_REQUEST_TEMPLATE.md
.. _`pull requests`: https://github.com/fat-forensics/fat-forensics/pulls
.. _issues: https://github.com/fat-forensics/fat-forensics/issues
.. _`issue templates`: https://github.com/fat-forensics/fat-forensics/tree/master/.github/ISSUE_TEMPLATE

Package Structure
=================

All the details of the package structure can be learnt from the
:ref:`API documentation <api_ref>`. However, we include a short summary below
for completeness.

Fairness
--------

.. autosummary::

   fatf.fairness
   fatf.fairness.data
   fatf.fairness.models
   fatf.fairness.predictions

Accountability
--------------

.. autosummary::

   fatf.accountability
   fatf.accountability.data
   fatf.accountability.models

Transparency
------------

.. autosummary::

   fatf.transparency
   fatf.transparency.data
   fatf.transparency.models
   fatf.transparency.predictions

Visualisations
--------------

.. autosummary::

   fatf.vis

Utilities
---------

.. autosummary::

   fatf.utils

   fatf.utils.data
   fatf.utils.models

   fatf.utils.array
   fatf.utils.distances
   fatf.utils.metrics
   fatf.utils.tools
   fatf.utils.testing

Package Resources
=================

The documentation is build on top of the Bootstrap_ (v4.3.1) and jQuery_
(3.4.1) libraries. The `custom theme`_ is based on Sphinx's `nature theme`_
(commit hash 1b1ebd2; 2nd January 2019).

The "Fork me on GitHub" ribbon is based on the CSS solution written by
codepo8_.

.. Previous ribbon: https://github.blog/2008-12-19-github-ribbons/

The package icons were created with `Amazon Alexa Icon Builder`_:

- FAT Forensics icon.

  * Style.

    + Size: Maximal size.
    + Type: Gradient.
    + RGB: 40, 40, 40.
    + Angle: 0.

  * Background.

    + Type: Gradient.
    + RGB: 185, 186, 70.
    + Angle: 0.

  * Border.

    + Type: Solid.
    + RGB: 166, 153, 134.

- Fairness, Accountability and Transparency icons.

  * Style.

    + Size: Maximal size.
    + Type: Gradient.
    + RGB: 40, 40, 40.
    + Angle: 0.

  * Background.

    + Type: Gradient.
    + RGB: 185, 186, 70.
    + Angle: 135.

  * Border.

    + Type: Solid.
    + RGB: 166, 153, 134.

.. _Bootstrap: https://getbootstrap.com/
.. _Jquery: https://jquery.com/
.. _`nature theme`: https://github.com/sphinx-doc/sphinx/blob/master/sphinx/themes/nature/static/nature.css_t
.. _codepo8: https://codepo8.github.io/css-fork-on-github-ribbon/
.. _`Amazon Alexa Icon Builder`: https://developer.amazon.com/docs/tools/icon-builder.html
