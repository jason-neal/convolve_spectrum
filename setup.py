# Setup file for equanimousoctotribble


import os
# from setuptools.command.test import test as TestCommand
# To use a consistent encoding
from codecs import open

from setuptools import find_packages, setup

long_description = " "


base_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)))

# https://www.reddit.com/r/Python/comments/3uzl2a/setuppy_requirementstxt_or_a_combination/
# Cannot use when spectrum_overload is a github directory
# with open('requirements/requirements.txt') as f:
#    requirements = f.read().splitlines()

setup(
    name='convolve_spectrum',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version="0.2",

    description='Spectrum convolution that handles uneven wavlengths.',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/jason-neal/convolve_spectrum',
    author='Jason Neal',
    author_email='jason.neal@astro.up.pt',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Natural Language :: English',
    ],

    # What does your project relate to?
    keywords=['astronomy'],

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=['contrib', 'docs', 'tests', 'test']),

    # test_suite=[],
    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #   py_modules=["my_module"],
    # py_modules=["spectrum/Spectrum"],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=["numpy", "matplotlib", "tqdm", "multiprocess"],
    # install_requires=[],
    setup_requires=['pytest-runner'],
    tests_require=['pytest', "hypothesis"],
    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extras_require={
        'dev': ['check-manifest'],
        'test': ['coverage', 'pytest', 'pytest-cov', 'python-coveralls', 'hypothesis'],
        'docs': ['sphinx >= 1.4', 'sphinx_rtd_theme',]
    },

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    # package_data={"spectrum_overload": ["data/*.fits"]},
    package_data={},

    data_files=[],

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
        #    'console_scripts': [
        #        'sample=sample:main',
        # ],
        'console_scripts': [],
    },
)
