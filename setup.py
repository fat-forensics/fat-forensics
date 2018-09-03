#! /usr/bin/env python
#
# Copyright (C) 2018 Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

from setuptools import (find_packages,
                        setup)

import fatf

DISTNAME = 'FAT-forensics'
PACKAGE_NAME = 'fatf'
VERSION = fatf.__version__
DESCRIPTION = ('A set of python modules to assess fairness, accountability and '
               'transparency of artificial intelligence techniques')
with open('README.rst') as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = 'Kacper Sokol'
MAINTAINER_EMAIL = 'k.sokol@bristol.ac.uk'
URL = 'https://anthropocentricai.github.io/{}'.format(DISTNAME)
DOWNLOAD_URL = 'https://pypi.org/project/{}/#files'.format(DISTNAME)
LICENSE = 'new BSD'
PACKAGES = find_packages(exclude=['*.tests', '*.tests.*', 'tests.*', 'tests'])
with open('requirements.txt') as f:
    INSTALL_REQUIRES = [l.strip() for l in f.readlines() if l]
INCLUDE_PACKAGE_DATA = True
ZIP_SAFE = False

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
                    long_description=LONG_DESCRIPTION,
                    include_package_data=INCLUDE_PACKAGE_DATA,
                    zip_safe=ZIP_SAFE,
                    packages=PACKAGES)

    setup(**metadata)

if __name__ == "__main__":
    setup_package()
