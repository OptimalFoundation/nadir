# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'nadir'
copyright = '2024, Bhavnick Minhas'
author = 'Bhavnick Minhas'
release = '0.1.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx_external_toc"]
external_toc_path = "_toc.yml"
external_toc_exclude_missing = False

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
html_theme_options = {
    "repository_url": "https://github.com/OptimalFoundation/nadir",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_download_button": True,
    "logo":{
        "image_light": "../logo.png",
        "image_dark": "../logo_dark.png",
    }
}
html_title = "Nadir"

