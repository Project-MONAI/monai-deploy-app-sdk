# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Handle ReadTheDocs.org build -------------------------------------------
import os

# on_rtd is whether we are on readthedocs.org
on_rtd = os.environ.get("READTHEDOCS", None) == "True"
if on_rtd:
    READTHEDOCS_PROJECT = os.environ.get("READTHEDOCS_PROJECT", "monai-deploy-app-sdk")
    READTHEDOCS_VERSION = os.environ.get("READTHEDOCS_VERSION", "latest")
    import subprocess

    subprocess.call(
        "/bin/bash -c 'source /home/docs/checkouts/readthedocs.org/user_builds/"
        f"{READTHEDOCS_PROJECT}/envs/{READTHEDOCS_VERSION}/bin/activate; ../../run setup read_the_docs'",
        shell=True,
    )
    subprocess.call(
        "/bin/bash -c 'source /home/docs/checkouts/readthedocs.org/user_builds/"
        f"{READTHEDOCS_PROJECT}/envs/{READTHEDOCS_VERSION}/bin/activate; ../../run setup_gen_docs'",
        shell=True,
    )

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import re
import sys

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
print(sys.path)


# -- Project information -----------------------------------------------------
project = "MONAI Deploy App SDK"
copyright = "2021 MONAI Consortium"
author = "MONAI Contributors"

# The full version, including alpha/beta/rc tags
from monai.deploy import __version__ as MONAI_APP_SDK_VERSION  # noqa: E402

short_version = MONAI_APP_SDK_VERSION.split("+")[0]
release = short_version
version = re.sub(r"(a|b|rc)\d+.*", "", short_version)

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build"]  # type: ignore

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
# source_suffix = {".rst": "restructuredtext", ".txt": "restructuredtext", ".md": "markdown"}

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinxcontrib.spelling",  # https://sphinxcontrib-spelling.readthedocs.io/en/latest/index.html
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    "sphinx.ext.ifconfig",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinxcontrib.bibtex",
    "myst_nb",
    "sphinx_copybutton",
    "sphinx_togglebutton",
    "sphinx_panels",  # https://sphinx-panels.readthedocs.io/en/latest/
    "ablog",
    "sphinxemoji.sphinxemoji",
    # https://myst-parser.readthedocs.io/en/latest/sphinx/use.html#automatically-create-targets-for-section-headers
    # "sphinx.ext.autosectionlabel",  <== don't need anymore from v0.13.0
    "sphinx_autodoc_typehints",
    "sphinxcontrib.mermaid",
]

autoclass_content = "both"
add_module_names = True
source_encoding = "utf-8"
# Prefix document path to section labels, to use:
# `path/to/file:heading` instead of just `heading`
# (https://www.sphinx-doc.org/en/master/usage/extensions/autosectionlabel.html)
# autosectionlabel_prefix_document = True
# autosectionlabel_maxdepth = 4
napoleon_use_param = True
napoleon_include_init_with_doc = True
set_type_checking_flag = True
autodoc_default_options = {
    # Make sure that any autodoc declarations show the right members
    "members": True,
    "inherited-members": True,
    "private-members": False,
    "show-inheritance": True,
}
autosummary_generate = True
numpydoc_show_class_members = False
napoleon_numpy_docstring = False  # force consistency, leave only Google


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# -- Options for HTML output -------------------------------------------------
#
# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "logo_link": "https://monai.io",
    "external_links": [
        {"url": "https://github.com/Project-MONAI/monai-deploy-app-sdk/issues/new/choose", "name": "SUBMIT ISSUE"}
    ],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/project-monai/monai-deploy-app-sdk",
            "icon": "fab fa-github-square",
        },
        {
            "name": "Twitter",
            "url": "https://twitter.com/projectmonai",
            "icon": "fab fa-twitter-square",
        },
    ],
    "collapse_navigation": False,
    "navigation_depth": 3,
    "show_toc_level": 2,
    "footer_items": ["copyright"],
    "navbar_align": "content",
    # https://pydata-sphinx-theme.readthedocs.io/en/latest/user_guide/sections.html#add-your-own-html-templates-to-theme-sections  # noqa
    "navbar_start": ["navbar-logo"],
}
html_context = {
    "github_user": "Project-MONAI",
    "github_repo": "monai-deploy-app-sdk",
    "github_version": "main",
    "doc_path": "docs/",
    "conf_py_path": "/docs/",
    "VERSION": version,
}
html_scaled_image_link = False
html_show_sourcelink = True
html_favicon = "../images/favicon.ico"
html_logo = "../images/MONAI-logo-color.png"
# Quicklinks idea is from https://github.com/pydata/pydata-sphinx-theme/issues/221#issuecomment-887622420
# "sidebar-nav-bs" is still need to add to avoid console error messages in the final home page.
html_sidebars = {
    "index": ["search-field", "sidebar-quicklinks", "sidebar-nav-bs"],
    "**": ["search-field", "sidebar-nav-bs"],
}
pygments_style = "monokai"


# -- Options for pydata-sphinx-theme -------------------------------------------------
#
# (reference: https://pydata-sphinx-theme.readthedocs.io/en/latest/user_guide/configuring.html)  # noqa
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["../_static"]
html_css_files = ["custom.css"]
html_title = f"{project} {version} Documentation"

# -- Options for sphinx-panels -------------------------------------------------
#
# (reference: https://sphinx-panels.readthedocs.io/en/latest/)
panels_add_bootstrap_css = False  # pydata-sphinx-theme already loads bootstrap css

# -- Options for linkcheck builder -------------------------------------------------
#
# Reference
# : https://www.sphinx-doc.org/en/master/usage/configuration.html?highlight=linkcheck#options-for-the-linkcheck-builder)
linkcheck_ignore = [r"^\/", r"^\.\."]


# -- Options for sphinx.ext.todo -------------------------------------------------
# (reference: https://www.sphinx-doc.org/en/master/usage/extensions/todo.html)
todo_include_todos = True


# -- Options for sphinxemoji.sphinxemoji -------------------------------------------------
#
# (reference: https://sphinxemojicodes.readthedocs.io/en/stable/#supported-codes)  # noqa
#
# https://myst-parser.readthedocs.io/en/latest/using/syntax-optional.html#markdown-figures  # noqa
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_admonition",
    "html_image",
    # "linkify",  # disable linkify to not confuse with the file name such as `app.py`
    "replacements",
    # "smartquotes",
    "substitution",
    "tasklist",
]
# https://myst-parser.readthedocs.io/en/latest/syntax/optional.html#syntax-header-anchors
myst_heading_anchors = 5


# -- Options for myst-nb -------------------------------------------------
#
# (reference: https://myst-nb.readthedocs.io/en/latest/)
# Prevent the following error
#     MyST NB Configuration Error:
#    `nb_render_priority` not set for builder: doctest
nb_render_priority = {"doctest": ()}
# Prevent creating jupyter_execute folder in dist
#  https://myst-nb.readthedocs.io/en/latest/use/execute.html#executing-in-temporary-folders  # noqa
execution_in_temp = True
jupyter_execute_notebooks = "off"


# -- Options for sphinxcontrib.spelling -------------------------------------------------
#
# (reference: https://sphinxcontrib-spelling.readthedocs.io/en/latest/customize.html)
spelling_word_list_filename = ["spelling_wordlist.txt"]
spelling_exclude_patterns = []  # type: ignore


# -- Setup for Sphinx --------------------------------------


# import subprocess
# def generate_apidocs(*args):
#     """Generate API docs automatically by trawling the available modules"""
#     module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "monai"))
#     output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "apidocs"))
#     apidoc_command_path = "sphinx-apidoc"
#     if hasattr(sys, "real_prefix"):  # called from a virtualenv
#         apidoc_command_path = os.path.join(sys.prefix, "bin", "sphinx-apidoc")
#         apidoc_command_path = os.path.abspath(apidoc_command_path)
#     print(f"output_path {output_path}")
#     print(f"module_path {module_path}")
#     subprocess.check_call(
#         [apidoc_command_path, "-e"]
#         + ["--implicit-namespaces"]  # /monai folder wouldn't have __init__.py so we need this option.
#         + ["-o", output_path]
#         + [module_path]
#         + [os.path.join(module_path, p) for p in exclude_patterns]
#     )

# Avoid "WARNING: more than one target found for cross-reference 'XXX': YYY, ZZZ"
# - https://github.com/sphinx-doc/sphinx/issues/4961
# - https://github.com/sphinx-doc/sphinx/issues/3866
from sphinx.domains.python import PythonDomain


class PatchedPythonDomain(PythonDomain):
    def resolve_xref(self, env, fromdocname, builder, typ, target, node, contnode):
        if "refspecific" in node:
            del node["refspecific"]
        return super(PatchedPythonDomain, self).resolve_xref(env, fromdocname, builder, typ, target, node, contnode)


def setup(app):
    # Hook to allow for automatic generation of API docs
    # before doc deployment begins.
    # app.connect("builder-inited", generate_apidocs)

    # Avoid "WARNING: more than one target found for cross-reference 'XXX': YYY, ZZZ
    app.add_domain(PatchedPythonDomain, override=True)
