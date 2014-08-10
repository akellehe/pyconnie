__author__ = "keats.kelleher@gmail.com (Andrew Kelleher)"

try:
  from setuptools import setup, find_packages
except ImportError:
  import distribute_setup
  distribute_setup.use_setuptools()
  from setuptools import setup, find_packages

setup(
    name='pyconnie',
    version='1.0',
    packages=find_packages(),
    author='Andrew Kelleher',
    author_email='keats.kelleher@gmail.com',
    description='An implementation of the ConNIe',
    long_description='An implementation of Leskovec et. al\'s algorithm for inferring latent social networks while exploiting the convexity of the problem',
    test_suite='tests',
    install_requires=[
        'numpy==1.8.2',
        'scipy==0.14.0'
    ],
    url='http://www.github.com/akellehe/pyconnie',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Information Analysis',
    ],
    keywords='pyconnie leskovec connie ConNIe convexity latent social network inference'
)
