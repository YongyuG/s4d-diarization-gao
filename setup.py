#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
from setuptools import setup, find_packages
import s4d

setup(
    name='s4d',
    version=s4d.__version__,
    packages=find_packages(),
    author="Sylvain MEIGNIER",
    author_email="s4d@univ-lemans.fr",
    description="S4D: SIDEKIT for Diarization",
    long_description=open('README.md').read(),
    include_package_data=True,
    url='https://projets-lium.univ-lemans.fr/s4d/',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)",
        "Operating System :: OS Independent",
        "Development Status :: 5 - Production/Stable",
        "Topic :: Scientific/Engineering",
    ],
)
