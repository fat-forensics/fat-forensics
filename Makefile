# makefile to simplify repetitive build commands

package:
	pip install -e .

dependancies:
	pip install -r requirements.txt

dev-dependancies:
	pip install -r requirements-dev.txt

doc-test:
	#get from alex

doc-build:
	$(MAKE) -C docs html #update from alex

test:
	pytest

code-coverage:
	pytest --cov=./ --cov-report xml

check-linting-pylint:
	pylint fatf/

check-linting-flake8:
	flake8 fatf/



# token:
# 	#update token when we have it
# 	CODECOV_TOKEN="754ec358-2c9e-4f15-8123-b44041c1f09b" 
