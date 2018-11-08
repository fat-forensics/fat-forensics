# makefile to simplify repetitive build commands

package:
	pip install -e .

dependencies:
	pip install -r requirements.txt

dev-dependencies:
	pip install -r requirements-dev.txt

doc-test:
	pytest --doctest-glob='*.rst'

doc-build:
	sphinx-apidoc -o doc/source/ fatf/ fatf/tests/ fatf/transform/tests fatf/metrics/tests/ fatf/analyse/tests
	sphinx-build -b html ./doc/source ./doc/build

test:
	pytest

code-coverage:
	pytest --cov=./ --cov-report=xml:coverage.xml

check-linting-pylint:
	pylint fatf/

check-linting-flake8:
	flake8 fatf/

render-readme:
	pandoc -t html README.rst -o README.html
