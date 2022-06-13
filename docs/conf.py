#!/usr/bin/env python
#
# Daf documentation build configuration file, created by
# sphinx-quickstart on Fri Jun  9 13:47:02 2017.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another
# directory, add these directories to sys.path here. If the directory is
# relative to the documentation root, use os.path.abspath to make it
# absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# TODO: Remove this monkey-patching if/when sphinx fixes https://github.com/sphinx-doc/sphinx/issues/10333
from sphinx.util import inspect
inspect.TypeAliasForwardRef.__repr__ = lambda self: self.name
inspect.TypeAliasForwardRef.__hash__ = lambda self: hash(self.name)

# -- General configuration ---------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
]

autoclass_content = "both"
default_role = "py:obj"
autodoc_member_order = 'bysource'
show_inheritance = True
autodoc_typehints_format = "short"
todo_include_todos = True
autodoc_type_aliases = {
  # NewType:
  "Vector": "daf.typing.vectors.Vector",
  "DenseInColumns": "daf.typing.dense.DenseInColumns",
  "DenseInRows": "daf.typing.dense.DenseInRows",
  "FrameInColumns": "daf.typing.frame.FrameInColumns",
  "FrameInRows": "daf.typing.frame.FrameInRows",
  "SparseInColumns": "daf.typing.frame.SparseInColumns",
  "SparseInRows": "daf.typing.frame.SparseInRows",
  # Union: (doesn't work...)
  "AnyData": "daf.typing.unions.AnyData",
  "AxisView": "daf.storage.views.AxisView",
  "Dense": "daf.typing.dense.Dense",
  "DType": "daf.typing.dtypes.DType",
  "DTypes": "daf.typing.dtypes.DTypes",
  "Frame": "daf.typing.frames.Frame",
  "Known1D": "daf.typing.unions.Known1D",
  "Known2D": "daf.typing.unions.Known2D",
  "Known": "daf.typing.unions.Known",
  "Matrix": "daf.typing.matrices.Matrix",
  "MatrixInColumns": "daf.typing.matrices.MatrixInColumns",
  "MatrixInRows": "daf.typing.matrices.MatrixInRows",
  "Proper1D": "daf.typing.unions.Proper1D",
  "Proper2D": "daf.typing.unions.Proper2D",
  "Proper": "daf.typing.unions.Proper",
  "ProperInColumns": "daf.typing.unions.ProperInColumns",
  "ProperInRows": "daf.typing.unions.ProperInRows",
  "Sparse": "daf.typing.sparse.Sparse",
}
autosectionlabel_prefix_document = True
nitpicky = True
nitpick_ignore = [
    ('py:class', 'abc.ABC'),
    ('py:class', 'anndata._core.anndata.AnnData'),
    ('py:class', 'daf.access.CALLABLE'),
    ('py:class', 'daf.storage.memory._MemoryReader'),
    ('py:class', 'daf.typing.frame.FrameInColumns'),
    ('py:class', 'daf.typing.frame.FrameInRows'),
    ('py:class', 'daf.typing.frame.SparseInColumns'),
    ('py:class', 'daf.typing.frame.SparseInRows'),
    ('py:class', 'daf.typing.layouts.ColumnMajor'),
    ('py:class', 'daf.typing.layouts.RowMajor'),
    ('py:class', '_dense.DenseInColumns'),
    ('py:class', '_dense.DenseInRows'),
    ('py:class', '_fake_sparse.spmatrix'),
    ('py:class', 'h5py._hl.group.Group'),
    ('py:class', 'np.dtype'),
    ('py:class', 'numpy.dtype'),
    ('py:class', 'numpy.ndarray'),
    ('py:class', 'pandas.core.frame.DataFrame'),
    ('py:class', 'pandas.core.series.Series'),
    ('py:class', 'scipy.sparse._base.spmatrix'),
    ('py:class', 'scipy.sparse.base.spmatrix'),
    ('py:class', 'scipy.sparse._csc.csc_matrix'),
    ('py:class', 'scipy.sparse.csc.csc_matrix'),
    ('py:class', 'scipy.sparse._csr.csr_matrix'),
    ('py:class', 'scipy.sparse.csr.csr_matrix'),
    ('py:class', 'typing.Union'),
    ('py:class', '_unions.AnyData'),
    ('py:class', '_unions.Known'),
    ('py:class', '_vectors.Vector'),
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = 'Daf'
copyright = "2022 Weizmann Institute of Science"
author = "Oren Ben-Kiki"

# The version info for the project you're documenting, acts as replacement
# for |version| and |release|, also used in various other places throughout
# the built documents.
#
# The short X.Y version.
version = "0.1.0-dev.1"
# The full version, including alpha/beta/rc tags.
release = version

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, ``todo`` and ``todoList`` produce output, else they produce nothing.
# todo_include_todos = True
# todo_link_only = True


# -- Options for HTML output -------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
#html_theme = 'alabaster'
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

if not on_rtd:  # only import and set the theme if we're building docs locally
    import sphinx_rtd_theme
    html_theme = 'sphinx_rtd_theme'
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Theme options are theme-specific and customize the look and feel of a
# theme further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
#html_static_path = ['_static']

html_logo = "logo.svg"


# -- Options for HTMLHelp output ---------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'daf'


# -- Options for LaTeX output ------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass
# [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'daf.tex',
     'Daf Documentation',
     'Oren Ben-Kiki', 'manual'),
]


# -- Options for manual page output ------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'daf',
     'Daf Documentation',
     [author], 1)
]


# -- Options for Texinfo output ----------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'daf',
     'Daf Documentation',
     author,
     'daf',
     'One line description of project.',
     'Miscellaneous'),
]



