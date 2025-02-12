Welcome to the ``estim8`` documentation!
========================================


.. image:: https://img.shields.io/pypi/v/estim8
   :target: https://pypi.org/project/estim8

.. image:: https://img.shields.io/badge/code%20on-Github-lightgrey
   :target: https://github.com/JuBiotech/estim8



``estim8`` is a Python package for parameter estimation and uncertainty quantification in dynamical biological models. It is designed to be easy to use, and to provide a consistent interface for a variety of parameter estimation and uncertainty quantification methods.

The framework features model definition using a broad range of third-party modeling tools comliant to the `FMI <https://fmi-standard.org/>`_ standard, such as the open sourc environment `OpenModelica <https://openmodelica.org/>`_.

It implements special features for common biotechnological applications, like e.g. fitting multiple experimental replicates.
Model optimization utilizes a variety of metaheuristic optimization algorithms, including the *generalized islands* approach from `pygmo <https://esa.github.io/pagmo2/>`_  as well as functions from `scipy.optimize <https://docs.scipy.org/doc/scipy/reference/optimize.html>`_.

Installation
======================
.. toctree::
   :caption: Installtion
   :maxdepth: 2


It is highly recommended to use a `conda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`_  or preferably a `mamba <https://github.com/mamba-org/mamba>`_  environment manager.

Installation from PyPi
----------------------
``estim8`` can easily be installed using ``pip``:

.. code-block:: bash

    pip install estim8

Please note that not all dependencies are packaged with the PyPI version. In order to use ``pygmo`` optimizers, you have to install the package manually, e.g. using a ``mamba`` environment manager by running

.. code-block:: bash

    mamba install -c conda-forge pygmo


When using federated setups on Windows, additionally install ``m2w64-toolchain`` using ``mamba``:


.. code-block:: bash

    mamba install -c conda-forge m2w64-toolchain


Development installation
------------------------
Download the `source code repository <https://github.com/JuBiotech/estim8>`_ to your computer, best by using `git <https://git-scm.com/>`_:

#.   Navigate to the directory of your computer where you want the repository to be located
#.   Open a terminal and run :code:`git clone https://github.com/JuBiotech/estim8.git`
#.   Change into the dowloaded directory :code:`cd estim8`


It is advised to create a fresh virtual environent:

.. code-block:: bash

    mamba create --name estim8
    mamba activate estim8
    mamba env update --name estim8 --file environment.yml



Tutorials
===================
.. toctree::
   :caption: Tutorials
   :maxdepth: 1

   notebooks/1. Modeling & Simulation
   notebooks/2. Experimental data and error modeling
   notebooks/3. Parameter estimation
   notebooks/4. Modeling experimental replicates
   notebooks/5. Parameter Identifiability - Profile Likelihood
   notebooks/6. Uncertainty Quantification - Monte Carlo Sampling
   notebooks/7. Federated worker architecture


API Reference
============

.. toctree::
   :caption: API Reference
   :maxdepth: 2

   modules/datatypes
   modules/error_models
   modules/estimator
   modules/generalized_islands
   modules/models
   modules/profile
   modules/objective
   modules/optimizers
   modules/utils
   modules/visualization
   modules/workers
