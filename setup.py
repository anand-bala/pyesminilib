from setuptools import setup

setup(
    cffi_modules=["esmini/_build_esmini.py:ffibuilder"],
)
