# simple makefile to simplify repetitive build commands

PYTEST ?= pytest

install:
	pip install -r requirements-dev.txt

test-code:
	$(PYTEST)

test-coverage:
	$(PYTEST) --cov=./ --cov-report xml

code-analysis:
	pylint fatf/
	flake8 fatf/

docs:
	$(MAKE) -C docs html

token:
	#update token when we have it
	CODECOV_TOKEN="754ec358-2c9e-4f15-8123-b44041c1f09b" 
