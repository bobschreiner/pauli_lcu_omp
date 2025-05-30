from setuptools import Extension, setup
import numpy # Import numpy

setup(
    ext_modules=[
        Extension(
            "pauli_lcu_module",
            ["src/c/python_bindings_omp.c"],
            include_dirs=["src/c", numpy.get_include()], # Add numpy's include directory
            extra_compile_args=['-O3', '-Xpreprocessor', '-fopenmp'],
            extra_link_args=['-lomp'],
        )
    ]
)
