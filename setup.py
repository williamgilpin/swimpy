from distutils.core import setup
# from setuptools import setup
#from swimpy.version import __version__
#version=__version__

setup(
    name='swimpy',
    version='0.1',
    description='Utilities for working with experimental fluid mechanics data',
    author='William Gilpin',
    author_email='firstnamelastname(as one word)@googleemailservice',
    requires=[ 'numpy', 'scipy', 'matplotlib','skimage'],
    py_modules=['config'],
    packages=['swimpy', ],
    package_data={'swimpy': ['*'],
    'swimpy.plt_fmt': ['*'],
    'swimpy.piv_series_tools': ['*'],
    'swimpy.boundary_contour_tools': ['*'],
    'swimpy.process_images': ['*'],
    'swimpy.process_vfields': ['*'],
    'swimpy.time_series_tools': ['*'],
    },
)