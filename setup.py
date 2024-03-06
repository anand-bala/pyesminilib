from setuptools import setup

setup(
    cffi_modules=["src/esmini/_build_esmini.py:ffibuilder"],
)
