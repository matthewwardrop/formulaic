from os import path
from setuptools import find_packages, setup

version_info = {}
with open("formulaic/_version.py") as version_file:
    exec(version_file.read(), version_info)

PWD = path.abspath(path.dirname(__file__))
with open(path.join(PWD, "README.md"), encoding="utf-8") as readme:
    long_description = readme.read()

setup(
    name="formulaic",
    version=version_info["__version__"],
    versioning="post",
    author=version_info["__author__"],
    author_email=version_info["__author_email__"],
    url="https://github.com/matthewwardrop/formulaic",
    description="An implementation of Wilkinson formulas.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    packages=find_packages(exclude=("tests",)),
    python_requires=">=3.7",
    setup_requires=["setupmeta"],
    install_requires=[
        "astor>=0.7.0",
        "interface_meta>=1.2",
        "numpy>=1.3",
        "pandas>=1.2",
        "scipy>=1.6",
        "wrapt>=1.0",
    ],
    extras_require={
        "arrow": ["pyarrow>=1"],
        "benchmarks": ["patsy", "rpy2", "uncertainties"],
        "calculus": ["sympy>=1.3,<1.10"],
        "test": [
            "black==22.1.0",
            "flake8==4.0.1",
            "pytest==6.2.5",
            "pytest-cov==3.0.0",
        ],
    },
)
