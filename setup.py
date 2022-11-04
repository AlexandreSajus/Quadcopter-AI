"""Setups the package for installation."""

from setuptools import setup


def get_requirements():
    """Load requirements from file."""
    requirements_file = open("requirements.txt")
    return requirements_file.readlines()


setup(install_requires=get_requirements())
