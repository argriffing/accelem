#!/usr/bin/env python
"""expectation maximization acceleration

"""

DOCLINES = __doc__.split('\n')

# This setup script is written according to
# http://docs.python.org/2/distutils/setupscript.html
#
# It is meant to be installed through github using pip.

from distutils.core import setup

import numpy as np


setup(
        name='accelem',
        version='0.1',
        description=DOCLINES[0],
        author='alex',
        url='https://github.com/argriffing/accelem/',
        download_url='https://github.com/argriffing/accelem/',
        packages=['accelem'],
        test_suite='nose.collector',
        package_data={'accelem' : ['tests/test_*.py']},
        )

