#! /usr/bin/env python
#
# Copyright (C) 2018 Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import re
from setuptools import find_packages, setup

import fatf

def dependencies_from_file(file_path):
    required = []
    with open(file_path) as f:
        for l in f.readlines():
            l_c = l.strip()
            # get not empty lines and ones that do not start with python
            # comment "#" (preceded by any number of white spaces)
            if l_c and not l_c.startswith('#'):
                required.append(l_c)
    return required

def get_dependency_version(dependency, list_of_dependencies):
    matched_dependencies = []

    reformatted_dependency = dependency.lower().strip()
    for dep in list_of_dependencies:
        dependency_version = re.split('~=|==|!=|<=|>=|<|>|===', dep)
        if dependency_version[0].lower().strip() == reformatted_dependency:
            matched_dependencies.append(dep)

    if not matched_dependencies:
        raise NameError(('{} dependency could not be found in the list of '
                         'dependencies.').format(dependency))

    return matched_dependencies

DISTNAME = 'FAT-Forensics'
PACKAGE_NAME = 'fatf'
VERSION = fatf.__version__
DESCRIPTION = ('A Python Toolbox for Algorithmic Fairness, Accountability and '
               'Transparency')
with open('README.rst') as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = 'Kacper Sokol'
MAINTAINER_EMAIL = 'k.sokol@bristol.ac.uk'
URL = 'https://anthropocentricai.github.io/{}'.format(DISTNAME)
DOWNLOAD_URL = 'https://pypi.org/project/{}/#files'.format(DISTNAME)
LICENSE = 'new BSD'
PACKAGES = find_packages(exclude=['*.tests', '*.tests.*', 'tests.*', 'tests'])
INSTALL_REQUIRES = dependencies_from_file('requirements.txt')
EXTRAS_REQUIRES_AUX = dependencies_from_file('requirements-aux.txt')
EXTRAS_REQUIRES_DEV = dependencies_from_file('requirements-dev.txt')
EXTRAS_REQUIRES_VIS = [
    get_dependency_version('matplotlib', EXTRAS_REQUIRES_AUX)]
EXTRAS_REQUIRES_ML = [
    get_dependency_version('scikit-learn', EXTRAS_REQUIRES_AUX)]
EXTRAS_REQUIRE = {
        'all': EXTRAS_REQUIRES_AUX,
        'dev': EXTRAS_REQUIRES_DEV,
        'ml': EXTRAS_REQUIRES_ML,
        'vis': EXTRAS_REQUIRES_VIS,
        }
# Python 3.5 and up but not commited to Python 4 support yet
PYTHON_REQUIRES = '~=3.5'
INCLUDE_PACKAGE_DATA = True
#ZIP_SAFE = False

def setup_package():
    metadata = dict(name=DISTNAME,
                    maintainer=MAINTAINER,
                    maintainer_email=MAINTAINER_EMAIL,
                    description=DESCRIPTION,
                    license=LICENSE,
                    url=URL,
                    download_url=DOWNLOAD_URL,
                    version=VERSION,
                    install_requires=INSTALL_REQUIRES,
                    extras_require=EXTRAS_REQUIRE,
                    long_description=LONG_DESCRIPTION,
                    include_package_data=INCLUDE_PACKAGE_DATA,
                    python_requires=PYTHON_REQUIRES,
                    #zip_safe=ZIP_SAFE,
                    packages=PACKAGES)

    setup(**metadata)

if __name__ == "__main__":
    setup_package()
