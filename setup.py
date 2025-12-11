from setuptools import find_packages, setup

setup(
    name='src',
    version='0.1.0',
    description='This project focuses on classifying six types of metal surface defects using a Convolutional Neural Network (CNN).',
    author='Abhishek Meghwal',
    packages=find_packages(where="src"),
    package_dir={"": "src"},  # <-- IMPORTANT FOR src/ LAYOUT
)
