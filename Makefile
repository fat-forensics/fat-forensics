install:
	pip install -r requirements-dev.txt

lint:
	pylint fatf/
	flake8 fatf/

tests:
	pytest
	pylint fatf/
	flake8 fatf/
	pytest --cov=./ --cov-report xml

docs:
	make -C docs html

token:
	#update token when we have it
	CODECOV_TOKEN="754ec358-2c9e-4f15-8123-b44041c1f09b" 
