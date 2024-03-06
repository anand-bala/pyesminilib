import platform
import sys

from setuptools import setup

cmdclass = {}

if platform.python_implementation() == "CPython":
    try:
        import wheel.bdist_wheel  # type: ignore

        class BDistWheel(wheel.bdist_wheel.bdist_wheel):
            def finalize_options(self) -> None:
                self.py_limited_api = f"cp3{sys.version_info[1]}"
                wheel.bdist_wheel.bdist_wheel.finalize_options(self)

        cmdclass["bdist_wheel"] = BDistWheel
    except ImportError:
        pass

setup(
    cmdclass=cmdclass,
    zip_safe=False,
    cffi_modules=["src/esmini/_build_esmini.py:ffibuilder"],
)
