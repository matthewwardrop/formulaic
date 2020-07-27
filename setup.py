from setuptools import find_packages, setup

version_info = {}
with open('formulaic/_version.py') as version_file:
    exec(version_file.read(), version_info)

test_deps = [
    'flake8',
    'pytest',
    'pytest-cov',
]

setup(
    name='formulaic',
    version=version_info['__version__'],
    versioning='post',
    author=version_info['__author__'],
    author_email=version_info['__author_email__'],
    url="https://github.com/matthewwardrop/formulaic",
    description='An implementation of Wilkinson formulas.',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],

    packages=find_packages(exclude=('tests',)),
    python_requires='>=3.6',
    setup_requires=['setupmeta'],
    install_requires=[
        'astor',
        'interface_meta>=1.2',
        'numpy',
        'pandas',
        'scipy',
        'wrapt',
    ],
    tests_require=test_deps,
    extras_require={
        'arrow': ['pyarrow'],
        'benchmarks': ['patsy', 'rpy2', 'uncertainties'],
        'calculus': ['sympy'],
        'test': test_deps,
    },

)
