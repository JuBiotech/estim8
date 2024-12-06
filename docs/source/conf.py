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
copyright = "2024, Tobias Latour, Daniel Strohmeier"
author = "Tobias Latour, Daniel Strohmeier"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx_rtd_theme",
    "myst_parser",
    "numpydoc",
    "nbsphinx",
    "nbsphinx_link",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
# html_static_path = ["_static"]
pygments_style = "sphinx"


# autodoc config
autodoc_mock_imports = ["pygmo", "m2w64-toolchain"]
