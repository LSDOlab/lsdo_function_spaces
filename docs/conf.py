# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
# import os
# import sys

# -- Project information -----------------------------------------------------

project = 'lsdo_function_spaces'
copyright = '2023-2026, Andrew Fletcher'
author = 'Andrew Fletcher'
version = '1.0'
release = '1.0.0'


# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx_rtd_theme",
    "autoapi.extension",
    "numpydoc",                 
    "sphinx_copybutton",            # allows copying code embedded in the docs rendered from .md or .ipynb files
    "myst_nb",                      # renders .md, .myst, .ipynb files
    "sphinx.ext.viewcode",          # adds the source code for classes and functions in auto generated api ref
    "sphinxcontrib.bibtex",         # for references and citations
]

# sphinxcontrib.bibtex options
bibtex_bibfiles = ['src/references.bib']

myst_title_to_header = True
myst_enable_extensions = ["dollarmath", "amsmath", "tasklist"]
nb_execution_mode = 'off'

# autoapi options
autoapi_dirs = ["../lsdo_function_spaces"]
autoapi_root = 'src/autoapi'
autoapi_type = 'python'
autoapi_file_patterns = ['*.py', '*.pyi']
autoapi_ignore = ['*test_*.py', '*/tests/*', '*debug_*.py', '*temp_*.py']
autoapi_options = [
    'members',
    'undoc-members',
    'show-inheritance',
    'show-module-summary',
    'special-members',
]
autoapi_add_toctree_entry = False
autoapi_member_order = 'groupwise'
autoapi_python_class_content = 'class'

root_doc = 'index'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['README.md', '_build', 'Thumbs.db', '.DS_Store', 'src/welcome.md']


# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'logo_only': False,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#2980B9',
    # toc options
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': True
}
