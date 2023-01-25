from setuptools import setup
from distutils.extension import Extension  
from Cython.Build import cythonize

setup(
    ext_modules = cythonize(
        [
            Extension("lapack_rankone", ["lapack_rankone.pyx"], libraries=["lapack"]),
        ],
    ),
)