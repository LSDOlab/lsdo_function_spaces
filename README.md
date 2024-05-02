# lsdo_function_spaces

<!---
[![Python](https://img.shields.io/pypi/pyversions/lsdo_function_spaces)](https://img.shields.io/pypi/pyversions/lsdo_function_spaces)
[![Pypi](https://img.shields.io/pypi/v/lsdo_function_spaces)](https://pypi.org/project/lsdo_function_spaces/)
[![Coveralls Badge][13]][14]
[![PyPI version][10]][11]
[![PyPI Monthly Downloads][12]][11]
-->

[![GitHub Actions Test Badge](https://github.com/LSDOlab/lsdo_function_spaces/actions/workflows/actions.yml/badge.svg)](https://github.com/lsdo_function_spaces/lsdo_function_spaces/actions)
[![Forks](https://img.shields.io/github/forks/LSDOlab/lsdo_function_spaces.svg)](https://github.com/LSDOlab/lsdo_function_spaces/network)
[![Issues](https://img.shields.io/github/issues/LSDOlab/lsdo_function_spaces.svg)](https://github.com/LSDOlab/lsdo_function_spaces/issues)


A template repository for LSDOlab projects

This repository serves as a package for creating functions from various function spaces. The intended purposes include parameterization,
creating functional representations of quantities, surrogate modeling, and education. A key aspect of this package is that it is implemented
using the Computational System Design Langauge (CSDL), which allows for automatic derivative computation making this a good package for 
optimization applications or any application where analytic derivative calculation can be helpful.

*README.md file contains high-level information about your package: it's purpose, high-level instructions for installation and usage.*

# Installation

## Installation instructions for users
For direct installation with all dependencies, run on the terminal or command line
```sh
pip install git+https://github.com/LSDOlab/lsdo_function_spaces.git
```
If you want users to install a specific branch, run
```sh
pip install git+https://github.com/LSDOlab/lsdo_function_spaces.git@branch
```

<!-- **Enabled by**: `packages=find_packages()` in the `setup.py` file. -->

## Installation instructions for developers
To install `lsdo_function_spaces`, first clone the repository and install using pip.
On the terminal or command line, run
```sh
git clone https://github.com/LSDOlab/lsdo_function_spaces.git
pip install -e ./lsdo_function_spaces
```

# For Developers
For details on documentation, refer to the README in `docs` directory.

For details on testing/pull requests, refer to the README in `tests` directory.

# License
This project is licensed under the terms of the **GNU Lesser General Public License v3.0**.
