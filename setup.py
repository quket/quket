from glob import glob
from os.path import basename
from os.path import splitext

from setuptools import setup
from setuptools import find_packages


def _requires_from_file(filename):
    return open(filename).read().splitlines()


setup(
    name="quket",
    version="0.9",
    license="Apache License",
    description="Quantum Unified Kernel and Emulator Toolbox",
    author="Takashi Tsuchimochi, et al.",
    email="tsuchimochi@gmail.com",
    url="https://qithub.com/tsuchimoc/quket.git",
    packages=find_packages(),
    install_requires=_requires_from_file('requirements.txt')
)
