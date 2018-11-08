# makefile to simplify repetitive build commands

# Set the default shell to /bin/bash (from /bin/sh) to support source command
#SHELL := /bin/bash

# Get environment variables if _envar.sh exists
-include _envar.sh

all: linting-pylint linting-flake8 test code-coverage doc-test
# Do all the tests

install:
	pip install -e .

dependencies:
	pip install -r requirements.txt

dependencies-dev:
	pip install -r requirements-dev.txt

doc-test:
	pytest --doctest-glob='*.rst'

doc-build:
	sphinx-apidoc -o doc/source/ fatf/ fatf/tests/ fatf/transform/tests fatf/metrics/tests/ fatf/analyse/tests
	sphinx-build -b html ./doc/source ./doc/build

test:
	pytest

code-coverage:
	pytest --cov=./ --cov-report=term-missing --cov-report=xml:coverage.xml

deploy-code-coverage:
# @ before the command suppresses printing it out, hence hides the token
ifndef CODECOV_TOKEN
	@echo 'CODECOV_TOKEN environment variable is NOT set'
	$(error CODECOV_TOKEN is undefined)
else
	@echo 'codecov -t $$CODECOV_TOKEN -f coverage.xml'
	@codecov -t $(CODECOV_TOKEN) -f coverage.xml
endif

linting-pylint:
	pylint fatf/

linting-flake8:
	flake8 fatf/

readme:
	pandoc -t html README.rst -o README.html
