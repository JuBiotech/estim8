# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# import sphinx_rtd_theme
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))


import estim8

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "estim8"
copyright = "2025, Forschungszentrum Jülich GmbH"
author = "Tobias Latour, Daniel Strohmeier"
release = estim8.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "numpydoc",
    "nbsphinx",
    "sphinx_copybutton",
    "sphinx_book_theme",
    "sphinxcontrib.mermaid",
    "myst_nb",
]
myst_enable_extensions = ["amsmath", "dollarmath"]
nb_execution_mode = "off"

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
pygments_style = "sphinx"

# autodoc config
autodoc_mock_imports = ["pygmo", "m2w64-toolchain"]
