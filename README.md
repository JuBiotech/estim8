[![PyPI version](https://img.shields.io/pypi/v/estim8)](https://pypi.org/project/estim8/)
[![pipeline](https://github.com/jubiotech/estim8/workflows/pipeline/badge.svg)](https://github.com/JuBiotech/estim8/actions)
[![coverage](https://codecov.io/gh/jubiotech/estim8/branch/main/graph/badge.svg)](https://app.codecov.io/gh/JuBiotech/estim8)
[![documentation](https://readthedocs.org/projects/estim8/badge/?version=latest)](https://estim8.readthedocs.io/en/latest)


# ``estim8``
`estim8` is a open source Python toolbox for parameter estimation of dynamic (bio)process models compliant to the _Functional Mockup Interface_ ([FMI](https://fmi-standard.org/)) standard.

It provides a high-level API for simulation and analysis of FMU models, employing ODE or DAE systems written and compiled from third-party software like e.g. the open source modeling environment of [OpenModelica](https://openmodelica.org/). `estim8` offers special functionality with respect to biotechnological applications, for example the modeling of experimental replicates or measurement noise.

For now, it relies on global meta-heuristic optimization algorithms from [`scipy`](https://scipy.org/) and the highly scalable approaches provided by [`pygmo`](https://esa.github.io/pygmo2/) to solve parameter estimation problems, and provides methods for uncertainty quantification.

`estim8` is build using a highly modular object-oriented architecture, making it easily extensible for custom flavoured implementations.

## Installation:
It is highly recommended to use a [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) or preferably a [mamba](https://github.com/mamba-org/mamba) environment manager.

### 1. Installation from PyPI

It is advised to create a fresh virtual environent using ``mamba`` (or ``conda`` alternatively):

```bash
    mamba create --name estim8 python==3.10
    mamba activate estim8
```



**_NOTE:_** Not all dependencies are packaged with the PyPI version. In order to use ``pygmo`` optimizers, the package needs to be installed manually. When using federated setups on Windows, additionally install ``m2w64-toolchain``. These packages need to be installed **before** installing ``estim8``!

```bash
    mamba install -c conda-forge pygmo m2w64-toolchain
```

``estim8`` can then easily be installed using ``pip``:

```bash
    pip install estim8
```


### 2 Development Installation
Download the [source code repository](https://github.com/JuBiotech/estim8) to your computer, best by using [`git`](https://git-scm.com/):
1. navigate to the directory of your computer where you want the repository to be located
2. open a terminal and run `git clone https://github.com/JuBiotech/estim8.git`
3. change into the dowloaded directory `cd estim8`


#### Setting up the environment
It is advised to create a fresh virtual environent:
```bash
conda create --name <env_name>
conda activate <env_name>
conda env update --name <env_name> --file environment.yml
```

#### Importing estim8
To get the most recent version, open a terminal in the repository and run:
```bash
git pull
```

In Python, add the following lines ath the top of your code. Don't forget to adjust the path.
 ```Python
 import sys
 sys.path.append('path/to/repo')
 ```

### 1.2 Installation from PyPI

```bash
pip install estim8
```


> **_NOTE:_**  Please note that not all dependencies are packaged with the PyPI version. In order to use `pygmo` optimizers, you have to install the package manually, e.g. using a `conda` environment manager by running
> ```bash
>conda install -c conda-forge pygmo
>```



## Documentation
Check out our [documentation](https://estim8.readthedocs.io/en/latest/), where we provide a series of example notebooks.

## Usage and Citing
``estim8`` is licensed under the [GNU Affero General Public License v3.0](https://github.com/JuBiotech/estim8/blob/main/license.md).

A citable publication is coming soon.
