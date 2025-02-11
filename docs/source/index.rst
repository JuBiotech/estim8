Welcome to the ``estim8`` documentation!
========================================


.. image:: https://img.shields.io/pypi/v/estim8
   :target: https://pypi.org/project/estim8

.. image:: https://img.shields.io/badge/code%20on-Github-lightgrey
   :target: https://github.com/JuBiotech/estim8

.. image:: https://zenodo.org/badge/358629210.svg
   :target: https://zenodo.org/badge/latestdoi/358629210

``estim8`` is a Python package for parameter estimation and uncertainty quantification in dynamical biological models comliant to the `FMI <https://fmi-standard.org/>`_ standard. It is designed to be easy to use, and to provide a consistent interface for a variety of parameter estimation and uncertainty quantification methods.

Installation
============
.. toctree::
   :maxdepth: 1


It is highly recommended to use a `conda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`_  or preferably a `mamba <https://github.com/mamba-org/mamba>`_  environment manager.

Installation from PyPi
----------------------
`estim8` can easily be installed using `pip`:
.. code-block:: bash

    pip install estim8

Please note that not all dependencies are packaged with the PyPI version. In order to use ``pygmo`` optimizers, you have to install the package manually, e.g. using a ``mamba`` environment manager by running

.. code-block:: bash

    mamba install -c conda-forge pygmo


Development installation
------------------------
Download the `source code repository <https://github.com/JuBiotech/estim8>`_ to your computer, best by using `git <https://git-scm.com/>`_:
1. navigate to the directory of your computer where you want the repository to be located
2. open a terminal and run :code:`git clone https://github.com/JuBiotech/estim8.git`
3. change into the dowloaded directory :code:`cd estim8`

It is advised to create a fresh virtual environent:
.. code-block:: bash
    conda create --name estim8
    conda activate estim8
    conda env update --name estim8 --file environment.yml



Tutorials
=========
.. toctree::
   :maxdepth: 1

   notebooks/1. Modeling & Simulation.ipynb
   notebooks/2. Experimental data and error modeling.ipynb
   notebooks/3. Parameter estimation.ipynb
   notebooks/4. Modeling experimental replicates.ipynb
   notebooks/5. Parameter Identifiability - Profile Likelihood.ipynb
   notebooks/6. Uncertainty Quantification - Monte Carlo Sampling.ipynb
   notebooks/7. Federated worker architecture.ipynb


API Reference
=============
.. toctree::
   :maxdepth: 2


   estim8


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
