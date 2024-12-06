===============
2 Installation
===============

We highly recommend using the package manager `Mamba <https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html#>`_ and to follow these steps:

    1.  Download and execute the latest `installer <https://github.com/conda-forge/miniforge/releases>`_ for your OS. Make sure to activate the option "Add Mambaforge to my PATH environment variable".

    2.  Open a terminal and create a fresh Python environment with:
        ::
            mamba create --name my_env_name Python=3.10 pygmo
    3.  Activate your environment:
        ::
            mamba activate my_env_name
    4.  Install ``estim8`` via
        ::
            pip install estim8
