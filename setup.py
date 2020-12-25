#!/usr/bin/env python3

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="brainstat",
    version="0.0.1",
    author="MNI-MICA Lab and MPI-CNG Lab",
    author_email="reinder.vosdewael@gmail.com and sheymaba@gmail.com",
    description="A toolbox for statistical analysis of neuroimaging data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MICA-LAB/BrainStat",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'pandas',
        'nibabel',
        'scipy',
        'nilearn',
        'numpy_groupies',
        'scikit_learn',
        'brainspace',
    ],
)
