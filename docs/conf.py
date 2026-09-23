# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

package_path = os.path.abspath('../')
sys.path.insert(0, package_path)
os.environ['PYTHONPATH'] = ';'.join((package_path, os.environ.get('PYTHONPATH', '')))

# -- Project information -----------------------------------------------------

project = 'ccd-diffusion-paper'
copyright = '2026, Roy T. Smart, Charles C. Kankelborg'
author = 'Roy T. Smart, Charles C. Kankelborg'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.viewcode',
    'jupyter_sphinx',
    'nbsphinx',
]
autosummary_generate = True  # Turn on sphinx.ext.autosummary
autosummary_imported_members = True
autosummary_ignore_module_all = False
autodoc_typehints = "description"

graphviz_output_format = 'png'
inheritance_graph_attrs = dict(rankdir='TB')

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# This pattern also affects html_static_path and html_extra_path.
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# Execute a notebook under docs/reports/ during the build unless it already
# holds its outputs, so the figures are regenerated from the code and data
# in the package, while a copy executed earlier against the same code and
# data (the docs workflow keeps one in its cache) is used as it is.
nbsphinx_execute = 'auto'
# The depletion-region section of the report refits every flat track on its
# own grid, ten minutes on a laptop and longer on a hosted runner, so no
# cell has a time limit of its own; the build job as a whole has one.
nbsphinx_timeout = -1

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

html_theme = 'pydata_sphinx_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# The track browser and the frame gallery are drawn by this script from
# files the build writes under _static/browser, see setup() below.
html_js_files = ['browser.js']
html_css_files = ['browser.css']

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/roytsmart/ccd-diffusion-paper",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
    ],
}

# https://github.com/readthedocs/readthedocs.org/issues/2569
master_doc = 'index'


intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'pylatex': ('https://jeltef.github.io/PyLaTeX/current/', None),
    'aastex': ('https://aastex.readthedocs.io/en/latest/', None),
}


def _browser_fingerprint():
    """
    A hash over everything the track browser's files are made from: the
    data of the package and the code that reads, fits, and renders it.
    """
    import hashlib
    import pathlib
    import ccd_diffusion.tracks

    package = pathlib.Path(ccd_diffusion.tracks.__file__).parent
    sources = sorted((package / 'data').glob('*')) + [
        package / name
        for name in ('_browser.py', '_extract.py', '_tracks.py', '_fit.py', '_depleted.py')
    ]
    digest = hashlib.sha256()
    for path in sources:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _export_browser(app):
    """
    Write every track and the rendered frames of each camera and campaign
    under _static/browser, where browser.js reads them, unless the files
    there were made from the same data and code, which a stamp beside them
    records. The frames are fetched from the archive, so a fresh export
    needs the network and takes minutes.
    """
    import pathlib
    import ccd_diffusion.tracks as tracks

    static = pathlib.Path(__file__).parent / '_static' / 'browser'
    stamp = static / 'stamp.txt'
    fingerprint = _browser_fingerprint()
    if (
        stamp.exists()
        and stamp.read_text() == fingerprint
        and (static / 'tracks.json').exists()
        and (static / 'frames' / 'frames.json').exists()
    ):
        print('track browser: the exported files are current, kept')
        return
    num = tracks.export_tracks(static / 'tracks.json')
    rendered = tracks.export_frames(static / 'frames')
    stamp.write_text(fingerprint)
    print(f'track browser: {num} tracks and {len(rendered)} frames exported')


def setup(app):
    app.connect('builder-inited', _export_browser)

