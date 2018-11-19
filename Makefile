# makefile to simplify repetitive build commands

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

# Do all the tests
all: linting-pylint linting-flake8 test-with-code-coverage doc-test

install:
	pip install -e .

install-dev:
	pip install --no-deps -e .

dependencies:
	pip install -r requirements.txt

dependencies-dev:
ifdef FATF_TEST_SCIPY
ifeq ($(FATF_TEST_SCIPY),'latest')
	pip install scipy
else
	pip install scipy==$(FATF_TEST_SCIPY)
endif
endif
ifdef FATF_TEST_NUMPY
ifeq ($(FATF_TEST_NUMPY),'latest')
	pip install numpy
else
	pip install numpy==$(FATF_TEST_NUMPY)
endif
endif
	pip install -r requirements-dev.txt

doc-test:
	pytest --doctest-glob='*.rst'

doc-build:
	sphinx-apidoc \
		-o \
			doc/source/ \
			fatf/ fatf/tests/ \
			fatf/transform/tests \
			fatf/metrics/tests/ \
			fatf/analyse/tests
	sphinx-build -b html doc temp/doc

test:
	pytest --junit-xml=temp/pytest_$(PYTHON_VERSION).xml

code-coverage:
	pytest --cov=./ --cov-report=term-missing --cov-report=xml:temp/coverage.xml

test-with-code-coverage:
	pytest \
		--junit-xml=temp/pytest_$(PYTHON_VERSION).xml --cov=./ \
		--cov-report=term-missing --cov-report=xml:temp/coverage.xml

deploy-code-coverage:
# @ before the command suppresses printing it out, hence hides the token
ifeq ($(TRAVIS_PULL_REQUEST),'false')
ifndef CODECOV_TOKEN
	@echo 'CODECOV_TOKEN environment variable is NOT set'
	$(error CODECOV_TOKEN is undefined)
else
	@echo 'codecov -t $$CODECOV_TOKEN -f temp/coverage.xml'
#	@codecov -t $(CODECOV_TOKEN) -f temp/coverage.xml
endif
else
	@echo 'Code coverage can only be submitted from a branch of the upstream repo'
	$(error TRAVIS_PULL_REQUEST is undefined)
endif

linting-pylint:
	pylint fatf/

linting-flake8:
	flake8 fatf/

readme:
	pandoc -t html README.rst -o temp/README.html

validate-travis:
	travis lint .travis.yml
