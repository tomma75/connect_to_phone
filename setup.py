# setup.py
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("find_matching_points.pyx"),
)