from setuptools import setup
from Cython.Build import cythonize

# Cython setup, stolen from beginner Cython tutorial on Cython website
setup(
    ext_modules=cythonize("./abustibia.pyx"),
)

setup(
    ext_modules=cythonize("./heuristics.pyx")
)

setup(
    ext_modules=cythonize("./abustibia2.pyx")
)
