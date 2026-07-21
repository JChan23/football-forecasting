from setuptools import setup, find_packages

setup(
    name='football-forecasting',
    version='1.0.0',
    author='Joshua Chan',
    author_email='',
    description=(
        'Probabilistic football match forecasting framework. '
        'Placed top 1% (45th/3,515) in Jump Trading Probability Cup 2026.'
    ),
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/JChan23/football-forecasting',
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=[
        'numpy>=1.24.0',
        'scipy>=1.10.0',
        'pandas>=2.0.0',
        'matplotlib>=3.7.0',
    ],
    extras_require={
        'dev': ['jupyter>=1.0.0', 'notebook>=7.0.0'],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Intended Audience :: Science/Research',
    ],
)
