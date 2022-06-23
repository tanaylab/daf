#!/usr/bin/env python

import os
import platform
from setuptools import Extension, setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = open("requirements.txt").read().split()
test_requirements = open("requirements_test.txt").read().split()
dev_requirements = open("requirements_dev.txt").read().split()

setup(
    author="Oren Ben-Kiki",
    author_email="oren@ben-kiki.org",
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Software Development :: Libraries",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
    ],
    description="Data in Axes in Formats",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    long_description_content_type="text/x-rst",
    include_package_data=True,
    keywords="daf",
    name="daf",
    packages=find_packages(include=["daf"]),
    test_suite="tests",
    tests_require=test_requirements,
    extras_require={"dev": dev_requirements},
    url="https://github.com/tanaylab/daf.git",
    version="0.1.0",
)
