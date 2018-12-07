import setuptools

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name='VayDiff',
    version='0.0.1',
    author='Abhimanyu Vasishth, Zheyu Wu, Yiming Xu',
    author_email=" ",
    description='Python package for Automatic Differentiation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(),
    keywords=['Python','Automatic differentiation'],
    url='https://github.com/cs207-group-11/cs207-FinalProject',
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
)
