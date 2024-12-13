[![PyPI version](https://img.shields.io/pypi/v/estim8)](https://pypi.org/project/estim8/)
[![pipeline](https://github.com/jubiotech/estim8/workflows/pipeline/badge.svg)](https://github.com/JuBiotech/estim8/actions)
[![coverage](https://codecov.io/gh/jubiotech/estim8/branch/main/graph/badge.svg)](https://app.codecov.io/gh/JuBiotech/estim8)
[![documentation](https://readthedocs.org/projects/estim8/badge/?version=latest)](https://estim8.readthedocs.io/en/latest)
[![DOI](https://zenodo.org/badge/713469041.svg)](https://zenodo.org/doi/)


`Estim8` is a tool for parameter estimation of differential-algebraic models (DAEs) using either model formulations in Modelica in form of Functional Mockup Units (FMUs). It uses global meta-heuristic optimization to solve parameter estimation problems, and provides methods for uncertainty quantification.

## 1. Installation:
It is highly recommended to use a [mamba](https://github.com/mamba-org/mamba) environment manager (commands are not changed).

### 1.1 Development Installation
Download the code repository to your computer, best by using `git`
1. navigate to the directory where you want the repository to be located
2. open a terminal and run `git clone https://github.com/JuBiotech/estim8.git`
3. change into the dowloaded directory `cd estim8`


#### Setting up the environment
1. conda create --name <env_name>
2. conda activate <env_name>
3. conda env update --name <env_name> --file environment.yml

#### Importing estim8
To get the most recent version, open a terminal in the repository and run
```bash
git pull
```

In Python, add the following lines to your code:
 ```Python
 import sys
 sys.path.append('path/to/repo')
 ```
