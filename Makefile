# Makefile simplifying repetitive dev commands

# Set the default shell to /bin/bash (from /bin/sh) to support source command
#SHELL := /bin/bash

$(mkdir -p temp)

ifndef TRAVIS_PYTHON_VERSION
	PYTHON_VERSION := $(shell python -V | grep -Eo '\d+.\d+.\d+')
else
	PYTHON_VERSION := $(TRAVIS_PYTHON_VERSION)
endif

# Get environment variables if _envar.sh exists
-include _envar.sh

.PHONY: all install install-dev dependencies dependencies-dev docs-html \
	docs-html-coverage docs-linkcheck docs-coverage test-docs test-notebooks \
	test code-coverage test-with-code-coverage deploy-code-coverage \
	linting-pylint linting-flake8 linting-yapf check-types build readme-gen \
	readme-preview validate-travis

all: \
	test-with-code-coverage \
	test-notebooks \
	test-docs \
	\
	docs-html \
	docs-linkcheck \
	docs-coverage \
	\
	check-types \
	linting-pylint \
	linting-flake8 \
	linting-yapf

install:
	pip install -e .

install-dev:
	pip install --no-deps -e .

dependencies:
	pip install -r requirements.txt

dependencies-dev:
ifdef FATF_TEST_NUMPY
ifeq ($(FATF_TEST_NUMPY),latest)
	pip install --upgrade numpy
else
	pip install numpy==$(FATF_TEST_NUMPY)
endif
endif
ifdef FATF_TEST_SCIPY
ifeq ($(FATF_TEST_SCIPY),latest)
	pip install --only-binary=scipy --upgrade scipy
else
	pip install --only-binary=scipy scipy==$(FATF_TEST_SCIPY)
endif
endif
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

#docs: Makefile
#	$(MAKE) -C docs $(filter-out $@,$(MAKECMDGOALS))
#	exit 0

# Catch-all unmatched targets -> do nothing (silently)
# This is needed for docs target as any argument for that command will be just
# another target for make. It makes it dangerous as `make docs all` will
# additionally execute all target for this make as well.
#%:
#	@:

# Check docs: references (-n -- nit-picky mode -- generates warnings for all
# missing references) and linkage (-W changes all warnings into errors meaning
# unlinked sources will cause the build to fail.)
docs-html:
	mkdir -p docs/_build
	mkdir -p docs/_static
	sphinx-build -M html docs docs/_build -nW -w docs/_build/nit-picky-html.txt
	cat docs/_build/nit-picky-html.txt
#	$(MAKE) -C docs html

docs-linkcheck:
	mkdir -p docs/_build/linkcheck
	sphinx-build -M linkcheck docs docs/_build
	cat docs/_build/linkcheck/output.txt
#	$(MAKE) -C docs linkcheck

docs-coverage:
	mkdir -p docs/_build/coverage
	sphinx-build -M coverage docs docs/_build
	cat docs/_build/coverage/python.txt
#	$(MAKE) -C docs html -b coverage  # Build html with docstring coverage report
#	$(MAKE) -C docs coverage

docs-doctest:
	sphinx-build -M doctest docs docs/_build
#	$(MAKE) -C docs doctest

docs-clean:
	sphinx-build -M clean docs docs/_build

# Do doctests only: https://github.com/pytest-dev/pytest/issues/4726
# Given that this is work-in-progress feature use docs-doctest instead
# (`-k 'not test_ and not Test'` is used as a hack -- no doctests in functions
# starting with `test_` and classes starting with `Test` will be found.)
test-docs:
	pytest \
		--doctest-glob='*.txt' \
		--doctest-glob='*.rst' \
		--doctest-modules \
		--ignore=docs/_build/ \
		-k 'not test_ and not Test' \
		docs/ \
		fatf/

test-notebooks:
	pytest \
		--nbval \
		examples/

test:
	pytest \
		--junit-xml=temp/pytest_$(PYTHON_VERSION).xml \
		fatf/

code-coverage:
	pytest \
		--cov-report=term-missing \
		--cov-report=xml:temp/coverage_$(PYTHON_VERSION).xml \
		--cov=fatf \
		fatf/

test-with-code-coverage:
	pytest \
		--junit-xml=temp/pytest_$(PYTHON_VERSION).xml \
		--cov-report=term-missing \
		--cov-report=xml:temp/coverage_$(PYTHON_VERSION).xml \
		--cov=fatf fatf/

deploy-code-coverage:
# @ before the command suppresses printing it out, hence hides the token
ifeq ($(TRAVIS_PULL_REQUEST),'false')
ifndef CODECOV_TOKEN
	@echo 'CODECOV_TOKEN environment variable is NOT set'
	$(error CODECOV_TOKEN is undefined)
else
	@echo 'codecov -t $$CODECOV_TOKEN -f temp/coverage_$(PYTHON_VERSION).xml'
#	@codecov -t $(CODECOV_TOKEN) -f temp/coverage_$(PYTHON_VERSION).xml
endif
else
	@echo 'Code coverage can only be submitted from a branch of the upstream repo'
	$(error TRAVIS_PULL_REQUEST is undefined)
endif

linting-pylint:
	pylint --rcfile=.pylintrc fatf/

linting-flake8:
	flake8 --config=.flake8 fatf/

linting-yapf:
	yapf --style .style.yapf -p -r -d -vv fatf/

check-types:
	mypy --config-file=.mypy.ini fatf/

build:
	python3 setup.py sdist bdist_wheel

readme-gen:
	pandoc -t html README.rst -o temp/README.html

readme-preview:
	restview README.rst

validate-travis:
	travis lint .travis.yml

validate-sphinx-conf:
	pylint --rcfile=.pylintrc -d invalid-name docs/conf.py
	flake8 --config=.flake8 docs/conf.py
	yapf --style .style.yapf -p -r -d -vv docs/conf.py
