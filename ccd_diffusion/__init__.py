"""
Create the figures and compile the LaTeX files for this article.
"""

from . import spie
from . import tracks
from ._acronyms import acronyms
from ._variables import variables
from ._authors import authors
from ._keywords import keywords
from . import figures
from . import tables
from . import sections
from ._document import document, pdf

__all__ = [
    "spie",
    "tracks",
    "acronyms",
    "variables",
    "authors",
    "keywords",
    "figures",
    "tables",
    "sections",
    "document",
    "pdf",
]
